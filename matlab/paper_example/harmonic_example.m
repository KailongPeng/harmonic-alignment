%%we use this script in the paper to make the corruption experiment.
%%我们在论文中使用这个脚本来做腐败实验。


close all;
clear all;
plt = true;
labels = loadMNISTLabels('train-labels.idx1-ubyte');
imgs = loadMNISTImages('train-images.idx3-ubyte');

N=1000;
NPs = 1; %how many percentages to run 运行多少个百分比
NWs = 1; %number of wavelets 小波的数量
Ns = 1; %number of iterations to run 运行的迭代次数
Ps = linspace(0,1,NPs);
%scale of wavelets  (eg Nf) to use 使用小波的比例（如Nf）。
Ws = [2 8 16 64];
imgs = imgs';
I = eye(784);
% kernel params 内核参数
k1 = 20;
a1 = 20;
pca1 = 100;
k2 = k1;
a2 = a1;
pca2 = pca1;
% Z = transformed
kZ = 10;
aZ = 10;
pcaZ = 0;
% diffusion time for final embedding 最终嵌入的扩散时间
dt = 1;
%
output = zeros(NPs, Ns, NWs, 2);  %store metrics in here 在这里存储指标
%%

for p = 1:NPs %对每一个p，获得
    p
    %build random matrix and replace prct of columns with I 构建随机矩阵，用I替换各列的prct
    pct = Ps(p) %此时Ps是1 p是1， pct是1    
    randomrot = orth(randn(784)); %random orthogonal rotation 随机正交旋转，这个的目的是构造一个数据库在另一个domain
    colReplace = randsample(784,floor(pct*784));
    randomrot(:,colReplace) = I(:,colReplace); %size(randomrot)=784 784 对角线全是1
    
    for iter = 1:Ns
        iter
        rng('shuffle');

    % sample two sets of digits from MNIST 从MNIST中抽取两组数字
        rs1 = randsample(length(labels), N); % 60000里面选1000个 图片数据
        rs2 = randsample(length(labels), N); % 60000里面选1000个 图片数据
        
    % slice the digits 切断数字
        x1 = imgs(rs1,:); %x1是1000个图片数据
        x2b = imgs(rs2,:); %x2b是另外1000个图片数据
        
    % transform x2
        x2 = (x2b*randomrot');
        x3 = [x1;x2];
        [u3, v3, L3] = diffusionCoordinates(x3,a1,k1,pca1); %this is for evaluating unaligned data.  You can also ploit this. 
    % slice the labels
        l1 = labels(rs1);
        l2 = labels(rs2);
        
    % run pca and classify 降维并且分类
        DM3  = u3*diag(exp(-(v3)));
        beforeprct = knnclassifier(DM3(1:N,:), l1, DM3(N+1:end,:), l2, 5);
        
    % construct graphs 
        [u1, v1, L1] = diffusionCoordinates(x1,a1,k1,pca1); %normalized L with diffusion coordinates for sample 1 ; a1=20 k1=20 pca1=100
        [u2, v2, L2] = diffusionCoordinates(x2,a2,k2,pca2); %... sample 2
        
    % get fourier coefficients
        x1hat = u1'*x1;
        x2hat = u2'*x2;
        
        for scale = 1:NWs
            %iterate over bandwidths
            Nf = Ws(scale);
            
            % build wavelets 构建小波，小波的输入是 v（特征值） Nf=2 2
            [we1, ~, ~] = build_wavelets(v1,Nf,2);
            [we2, ~, ~] = build_wavelets(v2,Nf,2);
            %we1 is the filter evaluated over the eigenvalues.  So we can
            %pointwise multiply each we1/2 by the fourier coefficients
            
            
            % evaluate wavelets over data in the spectral domain
            % stolen from gspbox, i have no idea how the fuck this works
            c1hat = bsxfun(@times, conj(we1), permute(x1hat,[1 3 2]));
            c2hat = bsxfun(@times, conj(we2), permute(x2hat,[1 3 2]));
            
            % correlate the spectral domain wavelet coefficients.
            blocks = zeros(size(c1hat,1), Nf, size(c2hat,1));
            for i = 1:Nf %for each filter, build a correlation
                blocks(:,i,:) = (squeeze(c1hat(:,i,:))*squeeze(c2hat(:,i,:))');
            end
            % construct transformation matrix
            M = squeeze(sum(blocks,2)); %sum wavelets up 
            [Ut,St,Vt] = randPCA(M, min(size(M))); %this is random svd

            St = St(St>0); %this is here from earlier experiments where I was truncating by rank.  
            % We can probably remove this.
            rk = length(St);
            Ut = Ut(:,1:rk);
            Vt = Vt(:,1:rk);


            T = Ut*Vt'; %the orthogonal transformation matrix
            
            % compute transformed data
            
            u1T =  u1* (T) ; %U1 in span(U2)
            T = T';

            u2T = u2 * (T); % U2 in span(U1)
            
            E = [u1 u1T; u2T u2];
            
            X = E *diag(exp(-dt.*([v1;v2])));
            [uZ, vZ, LZ] = diffusionCoordinates(X, aZ, kZ, pcaZ);
            
            Z = uZ*diag(exp(-vZ));
            afterprct = knnclassifier(Z(1:N,:), l1, Z(N+1:end,:), l2, 5);
            output(p, iter, scale, 1) = beforeprct;
            output(p,iter, scale, 2) = afterprct;
        end
    end
     
end
%%

%%
function prct = knnclassifier(x, lx, y, ly, k) %check the nearest neighbors and their associated labels. 使用knn来预测点的标签
    nns = knnsearch(x, y, 'k', k);
    prct = ly == mode(lx(nns),2);
    prct = sum(prct)/length(prct);
end

function [u,v,L] = diffusionCoordinates(x,a,k,npca) 
    %获得扩散的坐标 输入是 x（距离矩阵）；a（核的参数）k（核的参数）npca（取的pca的数量） a=20 k=20 npca=100
    %输出的 u   v   L
    
    %diffusion maps with normalized Laplacian 具有归一化拉普拉斯的扩散图
    %npca = 0 corresponds to NO pca
    [~, w] = alphakernel(x, 'a',a, 'k',k,'npca',npca);
    N = size(w,1);
    D = sum(w,2);
    w = w./(D*D'); %this is the anisotropic kernel 这就是各向异性的内核
    D = diag(sum(w,1));
    L = eye(N)-D^-0.5 * w * D^-0.5; %类似于Ms

    disp('svd L')
    [u,v,~] = randPCA(L,N); % randPCA 定义在 paper_example/randPCA.m， 目的是进行PCA的降维操作

    % [U,S,V] = randPCA(A,k,its,l) 
    % 输入。
    %  A--被近似的矩阵
    %  k--正在构建的近似的rank秩。 k必须是一个正整数，<=A的最小维度。并且默认为6
    %  its -- 块状Lanczos方法的完整迭代次数。其必须是一个非负的整数，默认为2
    %  l -- 块Lanczos迭代的块大小。l必须是一个>=k的正整数，并且默认为k+2
    % 
    %  输出（所有三个都需要）。
    %  U -- 对A进行等级-k逼近的USV'中的m x k矩阵。其中A是m x n；U的列是正交的。
    %  S--对A的秩-k近似USV'中的k x k矩阵。其中A是m x n；S的条目都是非负的。其唯一的非零项以非递增的顺序出现在 在对角线上
    %  V--对A的秩-k近似USV'中的n x k矩阵。其中A是m x n；V的列是正态的。
    
    %   outputs (all three are required):
    %   U -- m x k matrix in the rank-k approximation USV' to A,
    %        where A is m x n; the columns of U are orthonormal
    %   S -- k x k matrix in the rank-k approximation USV' to A,
    %        where A is m x n; the entries of S are all nonnegative,
    %        and its only nonzero entries appear in nonincreasing order
    %        on the diagonal
    %   V -- n x k matrix in the rank-k approximation USV' to A,
    %        where A is m x n; the columns of V are orthonormal
    
	% 因此输出的 也就是SVD的L=usv
    % u 是一个 m x k的矩阵 
    % v 是 对L的秩-k近似USV'中的k x k矩阵。其中A是m x n；S的条目都是非负的。其唯一的非零项以非递增的顺序出现在 在对角线上
    [ss,ix] = sort(diag(v));
    v = ss;
    u = u(:,ix);
    % trim trivial information 丢掉琐碎的信息
    u = u(:,2:end);
    v = v(2:end);
end

function [fe,Hk,mu] = build_wavelets(v,Nf,overlap) %构建小波
    lmax = max(v); %maximum laplacian eigenvalue 最大的拉普拉斯特征值
    k = @(x) sin(0.5 * pi * (cos(pi*x)).^2) .* (x>=-0.5 & x<= 0.5);
    %this is the itersine function 这是循环的函数

    Hk = cell(Nf,1); %we are gonna store some lambda functions in here 我们将在这里存储一些lambda函数

    scale = lmax/(Nf-overlap+1)*(overlap);

    mu = zeros(Nf,1);
    % this is translating the wavelets along the interval 0, lmax.  这是沿着0, lmax的区间对小波进行翻译。 
    for ii=1:Nf
        Hk{ii} = @(x) k(x/scale-(ii-overlap/2)/overlap)...
                    ./sqrt(overlap)*sqrt(2); %lambda functions for the spectral domain filters.. 谱域滤波器的lambda函数
        mu(ii) = (ii-overlap/2)/overlap * scale; %i think this is the mean of each filter 我想这是每个过滤器的平均值
    end
% response evaluation... this is the money  反应评估...... 
    fe=zeros(length(v),Nf);
    for ii=1:Nf
        fe(:,ii)=Hk{ii}(v); 
    end
end
