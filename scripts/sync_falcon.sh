#!/usr/bin/env bash
# Merge the latest Falcon upstream (https://github.com/falcon-neuro/falcon)
# into the current branch.
#
# Usage: scripts/sync_falcon.sh [upstream-branch]   (default: develop)
#
# Conflicts that are expected every sync are resolved automatically:
#   - upstream paths CLAS does not use (DROPPED) stay deleted
#   - files CLAS maintains on its own (OURS) keep the CLAS version
# Anything else (e.g. changes to falcon/ that overlap with CLAS patches) is
# left for manual resolution.
set -euo pipefail

REMOTE=falcon
URL=https://github.com/falcon-neuro/falcon.git
BRANCH=${1:-develop}

DROPPED=(
    .clang-format .clang-tidy .github .vscode/settings.json
    CHANGELOG.md analysis_options.yaml cspell.yaml docs falcon_gui
    install.sh scripts/format.sh scripts/setup scripts/tidy.sh
)
OURS=(README.md .gitignore .vscode/launch.json scripts/make.sh)

cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Uncommitted changes in the working tree; commit or stash them first." >&2
    exit 1
fi

git remote get-url "$REMOTE" >/dev/null 2>&1 || git remote add "$REMOTE" "$URL"
git fetch "$REMOTE"

if git merge-base --is-ancestor "$REMOTE/$BRANCH" HEAD; then
    echo "Already up to date with $REMOTE/$BRANCH."
    exit 0
fi

echo "Merging $REMOTE/$BRANCH:"
git log --oneline "HEAD..$REMOTE/$BRANCH"
git merge --no-commit --no-ff "$REMOTE/$BRANCH" || true
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
    echo "The merge could not start (see the git message above)." >&2
    exit 1
fi

# Keep unused upstream paths deleted (covers both conflicts and newly added files).
for p in "${DROPPED[@]}"; do
    git rm -r -q -f --ignore-unmatch -- "$p" >/dev/null
done

# Keep the CLAS version of independently maintained files.
for p in "${OURS[@]}"; do
    if [ -n "$(git diff --name-only --diff-filter=U -- "$p")" ]; then
        git checkout --ours -- "$p"
        git add -f -- "$p"
    fi
done

unmerged=$(git diff --name-only --diff-filter=U)
if [ -n "$unmerged" ]; then
    echo
    echo "Resolve these conflicts manually, then 'git add' them and 'git commit':"
    echo "$unmerged" | sed 's/^/  /'
    echo "(or 'git merge --abort' to undo the merge)"
    exit 1
fi

git commit --no-edit
echo "Merged $REMOTE/$BRANCH. Rebuild and test before merging into master."
