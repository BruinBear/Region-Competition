img = (imread('tree.jpg'));
figure;
imshow(img);
hold on;
x = 600+550*cos(0:0.1:2*pi)';
y = 600+550*sin(0:0.1:2*pi)';
hold on, plot([x;x(1)],[y;y(1)],'g')

alpha = 0.002;
beta = 0.0001;
gamma =50;
iterations = 100;

N = length(x);
a = gamma*(2*alpha+6*beta)+1;
b = gamma*(-alpha-4*beta);
c = gamma*beta;
P = diag(repmat(a,1,N));
P = P + diag(repmat(b,1,N-1), 1) + diag(   b, -N+1);
P = P + diag(repmat(b,1,N-1),-1) + diag(   b,  N-1);
P = P + diag(repmat(c,1,N-2), 2) + diag([c,c],-N+2);
P = P + diag(repmat(c,1,N-2),-2) + diag([c,c], N-2);
P = inv(P);


f = gradient(gradmag(img,30));



for ii = 1:iterations
   % Calculate external force
   coords = [x,y];
   fex = get_subpixel(f{1},coords,'linear');
   fey = get_subpixel(f{2},coords,'linear');
   % Move control points
   x = P*(x+gamma*fex);
   y = P*(y+gamma*fey);
   if mod(ii,5)==0
      plot([x;x(1)],[y;y(1)],'b')
   end
end
plot([x;x(1)],[y;y(1)],'r')


