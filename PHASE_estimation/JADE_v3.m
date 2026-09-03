function [IFvals, phasevals, zero_crossings, amplitudegram] = JADE_v3(x, tt, Ts, varargin)

%JADE_V3 Estimate instantaneous phase and frequency using DTW.
%
% This version retains the algorithmic structure of JADE_v2 while adding
% robust validation, consistent vector handling, safer breakpoint and
% zero-crossing handling, deterministic time grids, and a functional IF_N.
%
% Required inputs:
%   x         Signal to be analyzed
%   tt        Time vector for signal
%   Ts        Sampling period
%
% Optional inputs:
%   bpi       Breakpoint spacing. If > 1, evenly spaced breakpoints are used.
%   N         Polynomial order for within-cycle phase fitting. Default: 0.
%   smooth    Smooth input before zero-crossing detection (0/1). Default: 0.
%   ZC        A priori zero-crossing indices. Overrides detection.
%   templates A priori templates in a cell array. Requires ZC.
%   normalize Remove AM component by normalization (0/1). Default: 0.
%   IF_N      Polynomial order used to smooth final IF values. Default: 0.
%
% Outputs:
%   IFvals          Instantaneous frequency estimate.
%   phasevals       Instantaneous phase estimate.
%   zero_crossings  Zero-crossing indices used in estimation.
%   amplitudegram   Estimated template amplitude.
%
%   See also polyfitZero, scaleAmplitudes, getCorrectiongCurve.
%
%  Please cite:
%
%  J. C. Mouli, D. Anderson, A. Cicone. On the Instantaneous Phase and 
%  Frequency Estimation of a Non-stationary Signal. The JADE Algorithm. 
%  Submitted.
%  ArXiv https://arxiv.org/pdf/2604.14185
%

%% Input parsing
p = inputParser;

validScalarPosNum = @(v) isnumeric(v) && isscalar(v) && isfinite(v) && v > 0;
validNonnegativeInteger = @(v) isnumeric(v) && isscalar(v) && ...
    isfinite(v) && v >= 0 && v == round(v);
validBinary = @(v) isnumeric(v) && isscalar(v) && ismember(v, [0 1]);

addRequired(p, 'x', @(v) isnumeric(v) && isvector(v) && all(isfinite(v)));
addRequired(p, 'tt', @(v) isnumeric(v) && isvector(v) && all(isfinite(v)));
addRequired(p, 'Ts', validScalarPosNum);
addOptional(p, 'bpi', 1, validNonnegativeInteger);
addOptional(p, 'N', 0, validNonnegativeInteger);
addOptional(p, 'smooth', 0, validBinary);
addOptional(p, 'ZC', [], @(v) isempty(v) || ...
    (isnumeric(v) && isvector(v) && all(isfinite(v))));
addOptional(p, 'templates', {}, @iscell);
addOptional(p, 'normalize', 0, validBinary);
addOptional(p, 'IF_N', 0, validNonnegativeInteger);

parse(p, x, tt, Ts, varargin{:});

x = x(:).';
tt = tt(:).';

if numel(x) ~= numel(tt)
    error('JADE_v3:InputSizeMismatch', ...
        'x and tt must have the same number of elements.');
end

if numel(x) < 3
    error('JADE_v3:InputTooShort', ...
        'x and tt must contain at least three samples.');
end

if any(diff(tt) <= 0)
    error('JADE_v3:InvalidTimeVector', ...
        'tt must be strictly increasing.');
end

dtMedian = median(diff(tt));
if abs(dtMedian - Ts) > max(1e-10, 1e-6 * Ts)
    warning('JADE_v3:SamplingPeriodMismatch', ...
        ['Ts differs from the median spacing of tt. Ts is used for ', ...
         'phase/frequency calculations.']);
end

bpi = p.Results.bpi;
N = p.Results.N;
IF_N = p.Results.IF_N;

if bpi < 1
    error('JADE_v3:InvalidBreakpointSpacing', ...
        'bpi must be a positive integer.');
end

if isempty(p.Results.ZC) && ~isempty(p.Results.templates)
    error('JADE_v3:TemplatesRequireZC', ...
        'A priori templates require a priori zero-crossings (ZC).');
end

%% Optional amplitude normalization
if p.Results.normalize == 1
    [xs, ~] = scaleAmplitudes(x);
    fprintf('Removing AM using spline normalization scheme\n');
else
    xs = x;
end

%% Optional smoothing for zero-crossing detection
if p.Results.smooth == 1
    fprintf('smoothing input data\n');
    xs_sm = smoothdata(xs);
else
    xs_sm = xs;
end

%% Zero-crossing detection
if isempty(p.Results.ZC)
    % Detect sign changes, including transitions through exact zeros.
    crossingIdx = find(xs_sm(1:end-1) .* xs_sm(2:end) <= 0);

    % Collapse adjacent exact-zero crossings into one boundary.
    if ~isempty(crossingIdx)
        crossingIdx = crossingIdx([true, diff(crossingIdx) > 1]);
    end

    zero_crossings = crossingIdx(:).';

    if isempty(zero_crossings)
        error('JADE_v3:NoZeroCrossings', ...
            'No zero crossings were detected.');
    end
else
    fprintf('using provided zero-crossings\n');
    zero_crossings = p.Results.ZC(:).';

    if any(zero_crossings ~= round(zero_crossings))
        error('JADE_v3:InvalidZC', ...
            'ZC must contain integer sample indices.');
    end

    if any(zero_crossings < 1) || any(zero_crossings > numel(x))
        error('JADE_v3:InvalidZC', ...
            'ZC contains indices outside the input signal.');
    end

    if any(diff(zero_crossings) <= 0)
        error('JADE_v3:InvalidZC', ...
            'ZC must be strictly increasing.');
    end
end

if numel(zero_crossings) < 2
    error('JADE_v3:InsufficientZeroCrossings', ...
        'At least two zero crossings are required.');
end

numCycles = numel(zero_crossings) - 1;
templates = p.Results.templates;

if ~isempty(templates) && numel(templates) ~= numCycles
    error('JADE_v3:TemplateCountMismatch', ...
        ['The number of templates (%d) must equal the number of ', ...
         'zero-crossing intervals (%d).'], numel(templates), numCycles);
end

%% Main DTW loop
phasegram = zeros(1, 0);
amplitudegram = zeros(1, 0);

for i = 2:numel(zero_crossings)

    int = zero_crossings(i) - 1;
    prevint = zero_crossings(i-1);

    if int < prevint
        error('JADE_v3:InvalidCycle', ...
            'Zero-crossing indices do not define a valid cycle.');
    end

    cur_cycle = xs(prevint:int);
    distance = tt(int) - tt(prevint);

    if distance <= 0
        error('JADE_v3:InvalidCycleDuration', ...
            'Cycle duration must be positive.');
    end

    ft = 1 / (2 * distance);

    % Deterministic template time grid; no arbitrary +0.0001 correction.
    nTemplate = max(2, round(distance / Ts) + 1);
    templateTime = (0:nTemplate-1) * Ts;

    if templateTime(end) > distance
        templateTime(end) = distance;
    end

    if any(diff(templateTime) <= 0)
        templateTime = [0, distance];
    end

    %% Template construction
    if isempty(templates)

        signAtEnd = sign(xs_sm(int));
        if signAtEnd == 0 && int > prevint
            signAtEnd = sign(xs_sm(int-1));
        end

        if signAtEnd == 0
            error('JADE_v3:AmbiguousTemplateSign', ...
                'Could not determine template sign for cycle %d.', i-1);
        end

        unitTemplate = signAtEnd * sin(2*pi*templateTime*ft);

        % dtwCost expects the scalar amplitude as its first argument and
        % performs the multiplication internally.
        tempMatch = @(amp) dtwCost(amp, unitTemplate, cur_cycle);

        if signAtEnd > 0
            ampUpper = max(cur_cycle);
        else
            ampUpper = abs(min(cur_cycle));
        end

        ampUpper = max(ampUpper, eps);
        estAmp = fminbnd(tempMatch, 0, ampUpper);
        template = estAmp * unitTemplate;

    else

        template = templates{i-1};

        if ~isnumeric(template) || isempty(template) || ...
                any(~isfinite(template(:)))
            error('JADE_v3:InvalidTemplate', ...
                'Template %d must be a nonempty finite numeric vector.', i-1);
        end

        template = template(:).';
        estAmp = max(abs(template));

        if estAmp == 0
            warning('JADE_v3:ZeroTemplate', ...
                'Template %d has zero amplitude.', i-1);
        end
    end

    %% Dynamic time warping
    [~, ix, iy] = dtw(template, cur_cycle);

    ixn = iy(:).';
    iyn = ix(:).';

    if isempty(ixn) || isempty(iyn)
        error('JADE_v3:EmptyDTWPath', ...
            'DTW returned an empty warping path for cycle %d.', i-1);
    end

    iyn = iyn - iyn(1);

    %% Phase estimation
    if N > 0

        polynomial = polyfitZero(ixn, iyn, N);
        phi_est = zeros(1, N+1);

        for j = 1:(N+1)
            order = N + 1 - j;
            phi_est(j) = polynomial(j) * (ft / Ts^(order - 1));
        end

        sampledphase = polyval(phi_est, templateTime);
        sampledphase = sampledphase - sampledphase(1);

        if ~isempty(phasegram)
            sampledphase = sampledphase + phasegram(end);
        end

    else

        % Retain the DTW value immediately before each change in ixn,
        % matching the original JADE_v2 implementation.
        changeIdx = [find(diff(ixn) ~= 0), numel(ixn)];

        iyn_N = iyn(changeIdx);

        sampledphase = iyn_N * (ft * Ts);
        sampledphase = sampledphase - sampledphase(1);

        if ~isempty(phasegram)
            sampledphase = sampledphase + phasegram(end);
        end
    end

    curAmp = estAmp * ones(1, numel(sampledphase));

    phasegram = [phasegram, sampledphase]; %#ok<AGROW>
    amplitudegram = [amplitudegram, curAmp]; %#ok<AGROW>
end

if numel(phasegram) < 2
    error('JADE_v3:InsufficientPhaseData', ...
        'Insufficient phase data were generated.');
end

%% Cubic spline interpolation at breakpoints
if bpi > 1
    breakpoints = 1:bpi:numel(phasegram);

    if breakpoints(end) ~= numel(phasegram)
        breakpoints(end+1) = numel(phasegram);
    end
else
    breakpoints = zero_crossings(1:end-1) - ...
        (zero_crossings(1) - 1);

    breakpoints = breakpoints( ...
        breakpoints >= 1 & breakpoints <= numel(phasegram));

    breakpoints = unique(breakpoints, 'stable');
end

if numel(breakpoints) < 2
    error('JADE_v3:InsufficientBreakpoints', ...
        'At least two interpolation breakpoints are required.');
end

phase_at_bp = phasegram(breakpoints);

[phasecurve, ~, ~] = fit( ...
    breakpoints(:), phase_at_bp(:), 'cubicspline');

phasevals = feval(phasecurve, 1:breakpoints(end));
phasevals = phasevals(:).';

%% Instantaneous frequency
IFvals = diff(phasevals) / Ts;

% IF_N now actually modifies the returned IF estimate.
if IF_N > 0

    if IF_N >= numel(IFvals)
        error('JADE_v3:InvalidIFOrder', ...
            'IF_N must be smaller than the number of IF samples.');
    end

    fprintf('smoothing IF estimate by polynomial fitting\n');

    total_x = 1:numel(IFvals);
    IFpoly = polyfit(total_x, IFvals, IF_N);
    IFvals = polyval(IFpoly, total_x);
end

%% Match amplitudegram length to phase output
amplitudegram = amplitudegram( ...
    1:min(numel(amplitudegram), numel(phasevals)));

end


function C = dtwCost(x, template, cur_cycle)
    % x is the scalar amplitude. The multiplication is intentionally
    % performed here because this helper is used by fminbnd.
    C = dtw(x .* template, cur_cycle);
end


function [p,S,mu] = polyfitZero(x,y,degree)
% found at https://www.mathworks.com/matlabcentral/fileexchange/35401-polyfitzero

%% check args
% X & Y should be numbers
assert(isnumeric(x) && isnumeric(y),'polyfitZero:notNumeric', ...
    'X and Y must be numeric.')
dim = numel(x); % number of elements in X
% DEGREE should be scalar positive number between 1 & 10 inclusive
assert(isnumeric(degree) && isscalar(degree) && isfinite(degree) && ...
    degree>0 && degree<=10 && degree==round(degree), ...
    'polyfitZero:degreeOutOfRange', ...
    'DEGREE must be an integer between 1 and 10.')
% DEGREE must be less than number of elements in X & Y
assert(degree<dim && degree==round(degree), ...
    'polyfitZero:DegreeGreaterThanDim', 'DEGREE must be less than numel(X)')
% X & Y should be same size vectors
assert(isvector(x) && isvector(y) && dim==numel(y), ...
    'polyfitZero:vectorMismatch', 'X and Y must be vectors of the same length.')
%% solve
% convert X & Y to column vectors
x = x(:); y = y(:);
% Scale X.
% attribution: this is based on code from POLYFIT by The MathWorks Inc.
if nargout > 2
   mu = [0; std(x)];
   x = (x - mu(1))/mu(2);
end
% using pow() is actually as fast or faster than looping, same # of flops!
z = zeros(dim,degree);
for n = 1:degree
    z(:,n) = x.^(degree-n+1);
end
p = z\y; % solve
p = [p;0]; % set y-intercept to zero
%% error estimates
% attribution: this is based on code from POLYFIT by The MathWorks Inc.
if nargout > 1
    V = [z,ones(dim,1)]; % append constant term for Vandermonde matrix
    % Return upper-triangular factor of QR-decomposition for error estimates
    R = triu(qr(V,0));
    r = y - V*p;
    S.R = R(1:size(R,2),:);
    S.df = max(0,length(y) - (degree+1));
    S.normr = norm(r);
end
p = p'; % polynomial output is row vector by convention
end


function [signalScaled, instantaneousAmplitude] = scaleAmplitudes(signal)
	signalLength = length(signal);

	signalScaled = signal;
	correctionCurve = 1;

	for i = 1:3
		[tmpcorrectionCurve] = getCorrectiongCurve(signalScaled);

		if size(signalScaled, 1) > size(signalScaled, 2)
			signalScaled = signalScaled';
		end

		if size(correctionCurve, 1) > size(correctionCurve, 2)
			correctionCurve = correctionCurve';
		end

		signalScaled = signalScaled ./ tmpcorrectionCurve;
		correctionCurve = tmpcorrectionCurve .* correctionCurve;
	end


	instantaneousAmplitude = correctionCurve;

end


function [correctionCurve] = getCorrectiongCurve(signal)
	signalLength = length(signal);

	[yMax, xMax] = findpeaks( abs(signal) );

	[xMax, yMax] = linearSplineNearBoundary(xMax, yMax, signal);

	correctionCurve = spline(xMax, yMax, 1:signalLength);
   
end