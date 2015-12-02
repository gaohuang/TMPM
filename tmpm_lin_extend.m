function model=tmpm_lin_extend(Xl,Yl,Xu,Yu,paras)

if ~isfield(paras,'verbose')
    paras.verbose=1;
end
d=size(Xl,2);
X=[Xl;Xu];
Y=[Yl;Yu];
Xl1=Xl(Yl==1,:);
Xl2=Xl(Yl==-1,:);
Xu1=Xu(Yu==1,:);
Xu2=Xu(Yu==-1,:);
L=length(Yl);L1=sum(Yl==1);L2=sum(Yl==-1);
U=length(Yu);U1=sum(Yu==1);U2=sum(Yu==-1);
%%
lambda=paras.lambda;
N1=L1+lambda*U1;
N2=L2+lambda*U2;

Mu1=(sum(Xl1,1)+lambda*sum(Xu1,1))/N1;
Mu2=(sum(Xl2,1)+lambda*sum(Xu2,1))/N2;

X1c=bsxfun(@minus,[Xl1;Xu1],Mu1);
X2c=bsxfun(@minus,[Xl2;Xu2],Mu2);
Cov1=X1c'*bsxfun(@times,X1c,[ones(L1,1);lambda*ones(U1,1)])/N1;
Cov2=X2c'*bsxfun(@times,X2c,[ones(L2,1);lambda*ones(U2,1)])/N2;

% Cov1=cov(Xl(Yl==1,:),1)
% Cov2=cov(Xu(Yu==-1,:),1)

%% switch labels
Yu0=Yu;
for iter=1:50
    
    % train robust MPM
    [alfa(iter),a,b]=mpm_lin(Mu1',Mu2',Cov1,Cov2,0,paras.rho1,paras.rho2,0,1e-6,1e-6,50);
    out=Xu*a-b;
    
    if iter==1
        nn=1; % global counter
        acc_record(nn)=100*mean(Yu0==paras.Yu);
        kappa=abs(Mu1*a-Mu2*a)/(sqrt(a'*Cov1*a)+(a'*Cov2*a));
        kappa_reg=abs(Mu1*a-Mu2*a)/(sqrt(a'*Cov1*a+a'*paras.rho1*a)+(a'*Cov2*a));
        p_record(nn)=kappa^2/(1+kappa^2);
        p_reg_record(nn)=kappa_reg^2/(1+kappa_reg^2);
    end
    
    idx1=(Yu0==1);  % index for positively predicted points
    idx2=(Yu0==-1); % index for negtively predicted points
    ind1=find(idx1==true);
    ind2=find(idx2==true);
    
    count=0;
    while(min(out(idx1))<max(out(idx2)))
        count=count+1;
        
        [~,i]=min(out(idx1));
        [~,j]=max(out(idx2));
        
        ixk=ind1(i(1));
        iyl=ind2(j(1));
        xk=Xu(ixk,:);
        yl=Xu(iyl,:);
        
        % update the mean and covariance
        Mu10=Mu1;
        Mu20=Mu2;
        Cov10=Cov1;
        Cov20=Cov2;
        delta=lambda*(yl-xk);
        Mu1=Mu10+delta/N1;
        Mu2=Mu20-delta/N2;
        Offset1=(xk+yl)/2-Mu10;
        Offset2=(xk+yl)/2-Mu20;
        Cov1=Cov10+1/N1*(delta'*Offset1+Offset1'*delta)-delta'*delta/N1^2;
        Cov2=Cov20-1/N2*(delta'*Offset2+Offset2'*delta)-delta'*delta/N2^2;
   
%         Yu(ixk)=-1;
%         Yu(iyl)=1;
%         Xu1=Xu(Yu==1,:);
%         Xu2=Xu(Yu==-1,:);
%         X1c=bsxfun(@minus,[Xl1;Xu1],Mu1);
%         X2c=bsxfun(@minus,[Xl2;Xu2],Mu2);
%         Cov1a=X1c'*bsxfun(@times,X1c,[ones(L1,1);lambda*ones(U1,1)])/N1
%         Cov2a=X2c'*bsxfun(@times,X2c,[ones(L2,1);lambda*ones(U2,1)])/N2;
%         norm(Cov1-Cov1a)  
%         norm(Cov2-Cov2a)
        

        % check if the cost decreases
        flag1=Offset1*a;
        flag2=Offset2*a;
        
        if flag1<0&&flag2>0  % satisfies all the conditions, then simply switch the labels
            Yu0(ixk)=-1;
            Yu0(iyl)=1;
        else                 % otherwise, compute the cost explicitly
            cost0=(sqrt(a'*Cov10*a)+(a'*Cov20*a))/abs(Mu10*a-Mu20*a);
            cost1=(sqrt(a'*Cov1*a)+(a'*Cov2*a))/abs(Mu1*a-Mu2*a);
            if cost1<cost0      % if cost decreases, then switch
                Yu0(ixk)=-1;
                Yu0(iyl)=1;
            else                % otherwise, do not switch and remove them from the candidate pool
                Mu1=Mu10;
                Mu2=Mu20;
                Cov1=Cov10;
                Cov2=Cov20;
                if flag2<0
                    out(ixk)=Mu1*a-b;
                else
                    out(iyl)=Mu2*a-b;
                end
            end
        end
        
        % update the label vector and indices
        idx1(ixk)=0;
        idx1(iyl)=1;
        idx2(iyl)=0;
        idx2(ixk)=1;
        ind1=find(idx1==true);
        ind2=find(idx2==true);
        
        % record p and accuracy
        nn=nn+1;
        acc_record(nn)=100*mean(Yu0==paras.Yu);
        kappa=abs(Mu1*a-Mu2*a)/(sqrt(a'*Cov1*a)+(a'*Cov2*a));
        kappa_reg=abs(Mu1*a-Mu2*a)/(sqrt(a'*Cov1*a+a'*paras.rho1*a)+(a'*Cov2*a));
        p_record(nn)=kappa^2/(1+kappa^2);
        p_reg_record(nn)=kappa_reg^2/(1+kappa_reg^2);
        
%         % record p and accuracy
%         nn=nn+1;
%         Acc(nn)=100*mean(Yp0(L+1:end)==paras.Yu);
%         kappa=abs(mX*a-mY*a)/(sqrt(a'*covX*a)+(a'*covX*a));
%         p(nn)=kappa^2/(1+kappa^2);
    end
    model.switched_pair(iter)=count;
    if count==0
        disp(['Algorithm converges at iteration ', num2str(iter),'!'])
        break;
    elseif paras.verbose
        if isfield(paras,'Yu')
            pred=sign(Xu*a-b);
            disp(['Iter ', num2str(iter),': #switched points: ', num2str(2*count),...
                ' (alpha: ',num2str(alfa(iter)),'AccU: ',num2str(100*mean(pred==paras.Yu)),')'])
        else
            disp(['Iter ', num2str(iter),': #switched points: ', num2str(2*count),...
                ' (alpha: ',num2str(alfa(iter)),')'])
        end
    end
end
yu=Yu0;
model.a=a;
model.b=b;
model.alfa=alfa;
model.yu=yu;
model.iteration=iter;
% model.p1=100*p;
model.acc_record=acc_record;
model.p_record=100*p_record;
model.p_reg_record=100*p_reg_record;

