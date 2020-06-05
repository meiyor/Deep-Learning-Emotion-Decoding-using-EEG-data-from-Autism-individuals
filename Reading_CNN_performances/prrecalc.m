function [pr,re,acc,tn]=prrecalc(m,c)
a=diag(m);
acc=sum(a)./sum(sum(m));
if (all(size(m)==[2 1]))
    m(2,1)=0;
    m(2,2)=0;
end;
m
for (i=1:c)
 rer(i)=a(i)./sum(m(:,i));
end;
for(i=1:c)
  prr(i)=a(i)./sum(m(i,:));
end;
for(i=1:c)
  tnn(i)=sum(m(i,c-i+1))./sum(m(:,i));
end;
pr=mean(prr)
re=mean(rer) 
tn=mean(tnn);
