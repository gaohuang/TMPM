function [Xl,Yl,Xu,Yu,Xv,Yv,Xt,Yt,p] = LoadSet_ssl(set,split,path,preprocess)

path=[];
switch set
    case 1
        load([path,'ssl_g50c_10']);
    case 2
        load([path,'ssl_g241c_10']);
    case 3
        load([path,'ssl_breast_10']);
     case 4 
        load([path,'ssl_australian_10']);
    case 5
        load([path,'ssl_digit1_10']);
    case 6
        load([path,'ssl_pcmac_10']);
    case 7
        load([path,'ssl_textbook_10']);
    case 8
        load([path,'ssl_a9a_10']);
    case 9 
        load([path,'ssl_kdd500k_10']);
end

if preprocess==0
elseif preprocess==1
    xx=bsxfun(@times,xx,1./std(xx));
end

Xl=xx(idx_l(split,:),:);
Yl=yy(idx_l(split,:),:);
Xv=xx(idx_v(split,:),:);
Yv=yy(idx_v(split,:),:);
Xu=xx(idx_u(split,:),:);
Yu=yy(idx_u(split,:),:);
if exist('idx_t')==1
    Xt=xx(idx_t(split,:),:);
    Yt=yy(idx_t(split,:),:);
else
    Xt=xx(idx_l(split,:),:);
    Yt=yy(idx_l(split,:),:);
end


