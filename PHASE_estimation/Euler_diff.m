function [t1,x1,y1]=Euler_diff(t0,x0,y0,h,f1,f2)

m1=f1(t0,x0,y0);
m2=f2(t0,x0,y0);
x1=x0+m1*h;
y1=y0+m2*h;
t1=t0+h;

end
