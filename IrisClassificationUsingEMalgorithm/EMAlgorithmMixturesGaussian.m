function m = main()
  %Loading data into working space
  iris = importdata('iris.dat',' ');
  iris1 = iris(1:50, 1:4);
  iris2 = iris(51:100, 1:4);
  iris3 = iris(101:150, 1:4);

  train_mats1 = {iris1(11:50,:), iris2(11:50,:), iris3(11:50,:)};
  test_mats1 = {iris1(1:10,:), iris2(1:10,:), iris3(1:10,:)};
  train_mats2 = {[iris1(1:10,:); iris1(21:50,:)]
                [iris2(1:10,:); iris2(21:50,:)]
                [iris3(1:10,:); iris3(21:50,:)]};
  test_mats2 = {iris1(11:20,:), iris2(11:20,:), iris3(11:20,:)};
  train_mats3 = {[iris1(1:20,:); iris1(31:50,:)]
                [iris2(1:20,:); iris2(31:50,:)]
                [iris3(1:20,:); iris3(31:50,:)]};
  test_mats3 = {iris1(21:30,:), iris2(21:30,:), iris3(21:30,:)};
  train_mats4 = {[iris1(1:30,:); iris1(41:50,:)]
                [iris2(1:30,:); iris2(41:50,:)]
                [iris3(1:30,:); iris3(41:50,:)]};
  test_mats4 = {iris1(31:40,:), iris2(31:40,:), iris3(31:40,:)};
  train_mats5 = {iris1(1:40,:), iris2(1:40,:), iris3(1:40,:)};
  test_mats5 = {iris1(41:50,:), iris2(41:50,:), iris3(41:50,:)};

  train_mats = {train_mats1, train_mats2, train_mats3, train_mats4, train_mats5};
  test_mats  = {test_mats1, test_mats2, test_mats3, test_mats4, test_mats5};

  % Normalize [-1 1]
  for c = 1:4
    max_c = max(iris1(:,c));
    min_c = min(iris1(:,c));
    d = max_c - min_c;
    for r = 1:50
      iris1(r,c) = (iris1(r,c) - min_c) * 2 / d - 1;
    end
  end
  for c = 1:4
    max_c = max(iris2(:,c));
    min_c = min(iris2(:,c));
    d = max_c - min_c;
    for r = 1:50
      iris2(r,c) = (iris2(r,c) - min_c) * 2 / d - 1;
    end
  end
  for c = 1:4
    max_c = max(iris3(:,c));
    min_c = min(iris3(:,c));
    d = max_c - min_c;
    for r = 1:50
      iris3(r,c) = (iris3(r,c) - min_c) * 2 / d - 1;
    end
  end
  
   %Running EM algorithm for the 1st training mat
   disp('EM algorithm to find parameters of mixtures of two Gaussian distribution')
   [p,u, e] = EM_algorithm_mixture_of_Gaussian (train_mats[1])

end


function [p, u, e] = EM_algorithm_mixture_of_Gaussian (mat)
  u1 = mean(mat(1:20,:));
  u2 = mean(mat(21:40,:));
  u1_new = -1;
  u2_new = -1;

  while (u1_new ~= u1) | (u2_new ~= u2)
    if u1_new ~= -1 
      u1 = u1_new;
      u2 = u2_new;
    end

    nu = 0;
    de = 0;
    for x = 1:40
      v = mat(x,:);
      d1 = pdist([v;u1]);
      d2 = pdist([v;u2]);
      if d1 < d2
        nu = nu + v;
        de = de + 1;
      end
    end
    if de ~= 0
      u1_new = nu./de;
    else
      u1_new = u1;
    end

    nu = 0;
    de = 0;
    for x = 1:40
      v = mat(x,:);
      d1 = pdist([v;u1]);
      d2 = pdist([v;u2]);
      if d2 < d1
        nu = nu + v;
        de = de + 1;
      end
    end
    if de ~= 0
      u2_new = nu./de;
    else
      u2_new = u2;
    end
  end

  k1 = 0;
  k2 = 0;
  for x = 1:40
    v = mat(x,:);
    d1 = pdist([v;u1]);
    d2 = pdist([v;u2]);
    if d1 < d2
      k1 = k1 + 1;
      m1(k1,:) = v;    
    else
      k2 = k2 + 1;
      m2(k2,:) = v;    
    end
  end

  p = {(k1/(k1+k2)); (k2/(k1+k2))};
  u = {u1; u2};
  e = {cov(m1); cov(m1)};
end