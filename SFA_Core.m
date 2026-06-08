classdef SFA_Core
    % SFA_CORE Core estimation routines for Spatial SFA Latent Class Model
    % Author: Kai Du (kdu@uow.edu.au)
    % Version: 08062026
    % Reviewed:  
    %
    % Contains static methods for:
    %   - Parameter initialization
    %   - Parameter management
    %   - EM estimation

    methods (Static)
        
        function ParamMgr = initialize_parameter_manager(K, n_beta)
                        
            ParamMgr = struct();
            
            % === CONFIGURATION ===
            ParamMgr.config = struct();
            ParamMgr.config.K = K;
            ParamMgr.config.n_beta = n_beta;
            ParamMgr.config.n_params_per_class = 1 + n_beta + 2; 
            ParamMgr.config.n_total = K * ParamMgr.config.n_params_per_class + 2;             
            ParamMgr.config.param_names = generate_parameter_names(K, n_beta);
            
            % === VALIDATION RULES ===
            ParamMgr.validation = struct();
            ParamMgr.validation.rho_bounds = [0, 0.99];
            ParamMgr.validation.lambda_bounds = [0.01, 1e4];
            ParamMgr.validation.sigma2_v_bounds = [1e-6, 1e6];
            ParamMgr.validation.delta_bounds = [-15, 15];
            ParamMgr.validation.beta_bounds = [-Inf, Inf];
            ParamMgr.validation.em_tolerance = 1e-6;     
            ParamMgr.validation.em_max_iterations = 500;   
            ParamMgr.validation.n_starts = 5;
            
            % === SPATIAL STATIONARITY ===
            ParamMgr.spatial = struct();
            ParamMgr.spatial.rho_lower = ParamMgr.validation.rho_bounds(1);
            ParamMgr.spatial.rho_upper = ParamMgr.validation.rho_bounds(2);
            ParamMgr.spatial.lambda_W = [];
            
            % === METHOD HANDLES ===
            % These reference LOCAL functions at the end of this file
            ParamMgr.methods.pack = @(ps, pm) pack_parameters(ps, pm);
            ParamMgr.methods.unpack = @(tv, pm) unpack_parameters(tv, pm);
            ParamMgr.methods.extract_class = @(tv, k, pm) extract_class_parameters(tv, k, pm);
            ParamMgr.methods.insert_class = @(tv, tk, k, pm) insert_class_parameters(tv, tk, k, pm);
            ParamMgr.methods.get_bounds = @(pm) get_parameter_bounds(pm);
            ParamMgr.methods.validate = @(tv, pm) validate_parameters(tv, pm);
        end
        
        function ParamMgr = compute_spatial_bounds(ParamMgr, lambda_W)
            ParamMgr = compute_spatial_bounds_local(ParamMgr, lambda_W);
        end
        
        function init_params = initialise_from_moments(data, K)
            init_params = initialise_from_moments_local(data, K);
        end
        
        function [estimates, converged, omega, std_errors] = estimate_em(data, K, params, ParamMgr, n_starts)
            if nargin < 5, n_starts = ParamMgr.validation.n_starts; end
            [estimates, converged, omega, std_errors] = estimate_em_local(data, K, params, ParamMgr, n_starts);
        end
        
        function loglik = compute_loglik(theta, data, K, ParamMgr)
            loglik = compute_loglik_local(theta, data, K, ParamMgr);
        end
        
    end 
end 

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function param_names = generate_parameter_names(K, n_beta)
n_per_class = 1 + n_beta + 2; 
n_total = K * n_per_class + 2;
param_names = cell(n_total, 1);
idx = 1;
for k = 1:K
    param_names{idx} = sprintf('rho_%d', k); idx = idx + 1;
    for j = 1:n_beta
        param_names{idx} = sprintf('beta%d_%d', j, k); idx = idx + 1;
    end
    param_names{idx} = sprintf('lambda_%d', k); idx = idx + 1;
    param_names{idx} = sprintf('sigma2_v_%d', k); idx = idx + 1;
end
param_names{idx} = 'delta_2_const';
param_names{idx+1} = 'delta_2_slope';
end

function init_params = initialise_from_moments_local(data, K)
n = data.n;
T = data.T;
Y = data.Y;
X = data.X;
z = data.z;
n_beta = size(X, 3);

Y_vec = Y(:);
X_mat = zeros(n*T, n_beta);
for j = 1:n_beta
    X_mat(:, j) = reshape(X(:, :, j), [n*T, 1]);
end
beta_pool = (X_mat' * X_mat) \ (X_mat' * Y_vec);

resid_vec = Y_vec - X_mat * beta_pool;
resid_mat = reshape(resid_vec, [n, T]);

firm_mean_resid = mean(resid_mat, 2);
firm_std_resid = std(resid_mat, 0, 2);
cluster_features = [firm_mean_resid, firm_std_resid];

opts = statset('MaxIter', 500, 'Display', 'off');
[class_idx, ~] = kmeans(cluster_features, K, 'Replicates', 5, 'Options', opts);

init_params = struct();
init_params.rho = zeros(K, 1);
init_params.beta = cell(K, 1);
init_params.lambda = zeros(K, 1);
init_params.sigma2_v = zeros(K, 1);

for k = 1:K
    in_class_k = (class_idx == k);
    Y_k     = Y(in_class_k, :);
    X_k     = X(in_class_k, :, :);
    Y_k_vec = Y_k(:);
    n_k     = sum(in_class_k);
    X_k_mat = zeros(n_k*T, n_beta);
    for j = 1:n_beta
        X_k_mat(:, j) = reshape(X_k(:, :, j), [n_k*T, 1]);
    end
    init_params.beta{k} = (X_k_mat' * X_k_mat) \ (X_k_mat' * Y_k_vec);

    resid_k_vec = Y_k_vec - X_k_mat * init_params.beta{k};
    sigma2_total = var(resid_k_vec);
    init_params.sigma2_v(k) = 0.4 * sigma2_total;
    init_params.lambda(k) = sqrt(0.6 / 0.4);
    init_params.lambda(k) = max(min(init_params.lambda(k), 5.0), 0.2);
    init_params.rho(k) = 0.0;
end

init_params.delta = cell(K, 1);
if K == 2
    y_binary = double(class_idx == 2);
    z_var = z(:, 2);
    corr_val = corr(z_var, y_binary);
    delta_2_slope = corr_val;
    delta_2_slope = max(min(delta_2_slope, 2.0), -2.0);
    
    init_params.delta{1} = [0; 0];
    init_params.delta{2} = [0.5; delta_2_slope];
else
    init_params.delta{1} = zeros(2, 1);
    for k = 2:K, init_params.delta{k} = [0.5; 0.5]; end
end
end

function [best_estimates, best_converged, best_omega, best_std_errors] = estimate_em_local(data, K, params, ParamMgr, n_starts)
theta_init = ParamMgr.methods.pack(params, ParamMgr);
max_iter = ParamMgr.validation.em_max_iterations;
tol = ParamMgr.validation.em_tolerance;
n_params = length(theta_init);

all_theta = zeros(n_params, n_starts);
all_converged = false(1, n_starts);
all_loglik = -Inf(1, n_starts);
all_omega = cell(1, n_starts);

theta_swapped = swap_class_labels(theta_init, K, ParamMgr);
[lb, ub] = ParamMgr.methods.get_bounds(ParamMgr);
all_starts = generate_multi_starts(theta_init, theta_swapped, n_starts, K, ParamMgr, lb, ub);

parfor s = 1:n_starts
    theta_s = all_starts(:, s);

    converged_s = false;
    omega_s = zeros(data.n, K);
    for iter = 1:max_iter
        theta_old = theta_s;
        omega_s = compute_posterior_probs(theta_s, data, K, ParamMgr);

        for k_class = 1:K
            theta_k = ParamMgr.methods.extract_class(theta_s, k_class, ParamMgr);
            theta_k_new = update_theta_k(omega_s(:,k_class), data, theta_k, ParamMgr);
            theta_s = ParamMgr.methods.insert_class(theta_s, theta_k_new, k_class, ParamMgr);
        end
        delta_s = update_delta(omega_s, data.z, theta_s, K, ParamMgr);
        theta_s(end-1:end) = delta_s{2};

        if norm(theta_s - theta_old) < tol
            converged_s = true;
            break;
        end
    end
    omega_s = compute_posterior_probs(theta_s, data, K, ParamMgr);

    all_theta(:, s) = theta_s;
    all_converged(s) = converged_s;
    all_omega{s} = omega_s;
    if converged_s
        all_loglik(s) = compute_loglik_local(theta_s, data, K, ParamMgr);
    end
end

[~, best_idx] = max(all_loglik);

theta = all_theta(:, best_idx);
best_converged = all_converged(best_idx);
best_omega = all_omega{best_idx};

if best_converged
    obj_fun_hess = @(t) -compute_loglik_local(t, data, K, ParamMgr);
    H_obs = compute_numerical_hessian(obj_fun_hess, theta, ParamMgr);
    best_std_errors = compute_se_from_hessian(H_obs, theta, ParamMgr);
else
    best_std_errors = NaN(size(theta));
end
best_estimates = theta';
best_std_errors = best_std_errors';
end

function theta_k_new = update_theta_k(omega_k, data, theta_k_old, ParamMgr)
[lb_full, ub_full] = ParamMgr.methods.get_bounds(ParamMgr);
n_beta = size(data.X, 3);
n_per_class = 1 + n_beta + 2;  
lb = lb_full(1:n_per_class);
ub = ub_full(1:n_per_class);

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point', ...
    'MaxIterations', 500, 'OptimalityTolerance', 1e-5);
obj_fun = @(theta) -compute_weighted_loglik(theta, omega_k, data);

try
    [theta_k_new, ~, exitflag] = fmincon(obj_fun, theta_k_old, [],[],[],[], lb, ub, [], options);
    if exitflag <= 0, theta_k_new = theta_k_old; end
catch
    theta_k_new = theta_k_old;
end
end

function delta_new = update_delta(omega, z, theta, K, ParamMgr)
lb = ParamMgr.validation.delta_bounds(1);
ub = ParamMgr.validation.delta_bounds(2);
params = ParamMgr.methods.unpack(theta, ParamMgr);
delta_1 = params.delta{1};
delta_2_init = params.delta{2};

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point', ...
    'MaxIterations', 500, 'OptimalityTolerance', 1e-5);

obj_fun = @(d2) -sum(sum(omega .* log(compute_class_probs_matrix(z, {delta_1, d2}, K))));

try
    [delta_2_opt, ~, exitflag] = fmincon(obj_fun, delta_2_init, [],[],[],[], [lb;lb], [ub;ub], [], options);
    if exitflag <= 0, delta_2_opt = delta_2_init; end
catch
    delta_2_opt = delta_2_init;
end
delta_new = cell(K,1);
delta_new{1} = delta_1;
delta_new{2} = delta_2_opt;
end

function theta_swapped = swap_class_labels(theta, K, ParamMgr)

n_p = ParamMgr.config.n_params_per_class;
theta_swapped = theta;

theta_1 = theta(1:n_p);
theta_2 = theta(n_p+1:2*n_p);
theta_swapped(1:n_p) = theta_2;
theta_swapped(n_p+1:2*n_p) = theta_1;

delta_start = K * n_p + 1;
theta_swapped(delta_start:delta_start+1) = -theta(delta_start:delta_start+1);
end

function theta_p = perturb_theta(theta_base, scale, K, ParamMgr, lb, ub)

n_beta = ParamMgr.config.n_beta;
theta_p = theta_base;
idx = 1;
for k = 1:K

    theta_p(idx) = max(lb(idx), min(ub(idx), theta_base(idx) + scale * 0.2 * randn()));
    idx = idx + 1;

    for j = 1:n_beta
        theta_p(idx) = theta_base(idx) * (1 + scale * 0.3 * randn());
        idx = idx + 1;
    end

    lo_mult_l = max(0.1, 1 - scale * 0.5);  
    hi_mult_l = 1 + scale * 1.0;
    log_mult_l = log(lo_mult_l + (hi_mult_l - lo_mult_l) * rand());
    theta_p(idx) = max(lb(idx), min(ub(idx), theta_base(idx) + log_mult_l));
    idx = idx + 1;

    lo_mult_s = max(0.1, 1 - scale * 0.5);
    hi_mult_s = 1 + scale * 1.0;
    log_mult = log(lo_mult_s + (hi_mult_s - lo_mult_s) * rand());
    theta_p(idx) = max(lb(idx), min(ub(idx), theta_base(idx) + log_mult));
    idx = idx + 1;
end

theta_p(idx) = max(lb(idx), min(ub(idx), theta_base(idx) + scale * 2 * randn()));

theta_p(idx+1) = max(lb(idx+1), min(ub(idx+1), theta_base(idx+1) + scale * 0.05 * randn()));
end

function all_starts = generate_multi_starts(theta_init, theta_swapped, n_starts, K, ParamMgr, lb, ub)

n_params = length(theta_init);
all_starts = zeros(n_params, n_starts);
for s = 1:n_starts
    if s == 1
        all_starts(:, s) = theta_init;
    elseif s == 2
        all_starts(:, s) = theta_swapped;
    else
        scale = 1 + floor((s - 3) / 2) / 3;  
        if mod(s, 2) == 1  
            all_starts(:, s) = perturb_theta(theta_init, scale, K, ParamMgr, lb, ub);
        else  
            all_starts(:, s) = perturb_theta(theta_swapped, scale, K, ParamMgr, lb, ub);
        end
    end
end
end

function loglik = compute_loglik_local(theta, data, K, ParamMgr)
if ~ParamMgr.methods.validate(theta, ParamMgr), loglik = -1e10; return; end
params = ParamMgr.methods.unpack(theta, ParamMgr);
jacob = compute_jacobian_determinants(params.rho, data.lambda_W, K);

n = data.n; T = data.T;
Y = data.Y; X = data.X; W = data.W; z = data.z;

loglik = 0;
for i = 1:n
    p_ik = compute_class_probs(z(i,:)', params.delta);
    loglik_ik = zeros(K, 1);
    for k = 1:K
        loglik_ik(k) = (1/n) * jacob(k);
        for t = 1:T
            resid = compute_residuals(Y(:,t), X(:,t,:), W, params.rho(k), params.beta{k});
            loglik_ik(k) = loglik_ik(k) + compute_single_loglik(resid(i), params.lambda(k), params.sigma2_v(k));
        end
    end
    max_LL = max(loglik_ik);
    if ~isfinite(max_LL), loglik = -1e10; return; end
    loglik = loglik + log(p_ik' * exp(loglik_ik - max_LL)) + max_LL;
end
end

function omega = compute_posterior_probs(theta, data, K, ParamMgr)
params = ParamMgr.methods.unpack(theta, ParamMgr);
n = data.n; T = data.T;
jacob = compute_jacobian_determinants(params.rho, data.lambda_W, K);
omega = zeros(n, K);

for i = 1:n
    p_ik = compute_class_probs(data.z(i,:)', params.delta);
    loglik_ik = zeros(K, 1);
    for k = 1:K
        loglik_ik(k) = (1/n)*jacob(k);
        for t = 1:T
            resid = compute_residuals(data.Y(:,t), data.X(:,t,:), data.W, params.rho(k), params.beta{k});
            loglik_ik(k) = loglik_ik(k) + compute_single_loglik(resid(i), params.lambda(k), params.sigma2_v(k));
        end
    end
    max_LL = max(loglik_ik);
    w_exp = p_ik .* exp(loglik_ik - max_LL);
    omega(i,:) = w_exp / sum(w_exp);
end
end

function loglik = compute_weighted_loglik(theta_k, omega_k, data)
n_beta = size(data.X, 3);
rho = theta_k(1); beta = theta_k(2:1+n_beta); lambda = exp(theta_k(2+n_beta)); sigma2_v = exp(theta_k(3+n_beta));  
jacob = compute_jacobian_determinant(rho, data.lambda_W);
n = data.n; T = data.T;
loglik = 0;
for i = 1:n
    if omega_k(i) > 1e-10
        LL_i = (1/n)*jacob;
        for t = 1:T
            resid = compute_residuals(data.Y(:,t), data.X(:,t,:), data.W, rho, beta);
            LL_i = LL_i + compute_single_loglik(resid(i), lambda, sigma2_v);
        end
        loglik = loglik + omega_k(i) * LL_i;
    end
end
end

function epsilon = compute_residuals(Y_t, X_t, W, rho, beta)
n_beta = length(beta);
X_beta = zeros(size(Y_t));
for j = 1:n_beta
    X_beta = X_beta + X_t(:,j) * beta(j);
end
epsilon = Y_t - rho * W * Y_t - X_beta;
end

function ll = compute_single_loglik(resid_i, lambda, sigma2_v)
sigma2 = (1 + lambda^2) * sigma2_v;
sigma = sqrt(sigma2);
pdf = -0.5 * log(2*pi*sigma2) - resid_i^2/(2*sigma2);
cdf = log(max(normcdf(-lambda*resid_i/sigma), 1e-10));
ll = pdf + cdf;
end

function jacob = compute_jacobian_determinants(rho, lam, K)
jacob = zeros(K,1);
for k = 1:K, jacob(k) = compute_jacobian_determinant(rho(k), lam); end
end

function jacob = compute_jacobian_determinant(rho, lam)
eigvals = 1 - rho * lam;
if any(eigvals <= 0), jacob = -Inf; return; end
jacob = sum(log(eigvals));
end

function P = compute_class_probs_matrix(z, delta_cell, K)
n = size(z,1);
eta = zeros(n, K);
for k = 1:K, eta(:, k) = z * delta_cell{k}; end
eta_max = max(eta, [], 2);
exp_eta = exp(eta - eta_max);
P = exp_eta ./ sum(exp_eta, 2);
end

function p_ik = compute_class_probs(z_i, delta)
K = length(delta);
eta = zeros(K,1);
for k = 1:K, eta(k) = z_i' * delta{k}; end
eta_max = max(eta);
exp_eta = exp(eta - eta_max);
p_ik = exp_eta / sum(exp_eta);
end

function H = compute_numerical_hessian(fun, theta, ParamMgr)
n = length(theta);
H = NaN(n, n);
epsilon = max(1e-5 * abs(theta), 1e-7);
[lb, ub] = ParamMgr.methods.get_bounds(ParamMgr);

try
    f0 = fun(theta);
    for i = 1:n
        step_i = min(epsilon(i), min((ub(i)-theta(i))/2, (theta(i)-lb(i))/2));
        if step_i < 1e-10, continue; end
        
        theta_p = theta; theta_p(i) = theta(i)+step_i;
        theta_m = theta; theta_m(i) = theta(i)-step_i;
        H(i,i) = (fun(theta_p) - 2*f0 + fun(theta_m))/(step_i^2);
        
        for j = i+1:n
            step_j = min(epsilon(j), min((ub(j)-theta(j))/2, (theta(j)-lb(j))/2));
            if step_j < 1e-10, continue; end
            
            theta_pp = theta; theta_pp(i)=theta_pp(i)+step_i; theta_pp(j)=theta_pp(j)+step_j;
            theta_mm = theta; theta_mm(i)=theta_mm(i)-step_i; theta_mm(j)=theta_mm(j)-step_j;
            theta_pm = theta; theta_pm(i)=theta_pm(i)+step_i; theta_pm(j)=theta_pm(j)-step_j;
            theta_mp = theta; theta_mp(i)=theta_mp(i)-step_i; theta_mp(j)=theta_mp(j)+step_j;
            
            val = (fun(theta_pp) - fun(theta_pm) - fun(theta_mp) + fun(theta_mm)) / (4*step_i*step_j);
            H(i,j) = val; H(j,i) = val;
        end
    end
catch
    H = NaN(n,n);
end
end

function se = compute_se_from_hessian(H, theta, ParamMgr)
se = NaN(size(theta));
try
    [lb, ub] = ParamMgr.methods.get_bounds(ParamMgr);
    tol = 1e-6;
    at_boundary = ((theta - lb) < tol) | ((ub - theta) < tol);
    
    H_inv = H \ eye(length(theta));
    diag_var = diag(H_inv);
    for i = 1:length(theta)
        if ~at_boundary(i) && diag_var(i) > 0
            se(i) = sqrt(diag_var(i));
        end
    end
catch
end
end

function theta = pack_parameters(params, ParamMgr)
K = ParamMgr.config.K;
n_beta = ParamMgr.config.n_beta;
theta = zeros(ParamMgr.config.n_total, 1);
idx = 1;
for k = 1:K
    theta(idx) = params.rho(k); idx = idx + 1;
    theta(idx:idx+n_beta-1) = params.beta{k}; idx = idx + n_beta;
    theta(idx) = log(params.lambda(k)); idx = idx + 1;   
    theta(idx) = log(params.sigma2_v(k)); idx = idx + 1; 
end
theta(idx:idx+1) = params.delta{2};
end

function params = unpack_parameters(theta, ParamMgr)
theta = theta(:); 
K = ParamMgr.config.K;
n_beta = ParamMgr.config.n_beta;
params = struct();
params.rho = zeros(K,1); params.beta = cell(K,1); params.lambda = zeros(K,1); params.sigma2_v=zeros(K,1);
idx = 1;
for k = 1:K
    params.rho(k) = theta(idx); idx = idx + 1;
    params.beta{k} = theta(idx:idx+n_beta-1); idx = idx + n_beta;
    params.lambda(k) = exp(theta(idx)); idx = idx + 1;  
    params.sigma2_v(k) = exp(theta(idx)); idx = idx + 1; 
end
params.delta{1} = [0;0];
params.delta{2} = theta(idx:idx+1);
end

function tk = extract_class_parameters(theta, k, ParamMgr)
n_p = ParamMgr.config.n_params_per_class;
tk = theta((k-1)*n_p+1 : k*n_p);
tk = tk(:); 
end

function theta = insert_class_parameters(theta, tk, k, ParamMgr)
n_p = ParamMgr.config.n_params_per_class;
theta((k-1)*n_p+1 : k*n_p) = tk;
end

function [lb, ub] = get_parameter_bounds(ParamMgr)
K = ParamMgr.config.K;
n_beta = ParamMgr.config.n_beta;
n_total = ParamMgr.config.n_total;
lb = -Inf(n_total,1); ub = Inf(n_total,1);
idx = 1;
for k = 1:K
    lb(idx) = ParamMgr.spatial.rho_lower; ub(idx) = ParamMgr.spatial.rho_upper; idx=idx+1;
    lb(idx:idx+n_beta-1) = ParamMgr.validation.beta_bounds(1);
    ub(idx:idx+n_beta-1) = ParamMgr.validation.beta_bounds(2); idx=idx+n_beta;
    lb(idx) = log(ParamMgr.validation.lambda_bounds(1)); ub(idx) = log(ParamMgr.validation.lambda_bounds(2)); idx=idx+1;   
    lb(idx) = log(ParamMgr.validation.sigma2_v_bounds(1)); ub(idx) = log(ParamMgr.validation.sigma2_v_bounds(2)); idx=idx+1; 
end
lb(end-1:end) = ParamMgr.validation.delta_bounds(1);
ub(end-1:end) = ParamMgr.validation.delta_bounds(2);
end

function ParamMgr = compute_spatial_bounds_local(ParamMgr, lambda_W)
lambda_max = max(lambda_W);

ParamMgr.spatial.rho_lower = 0;
ParamMgr.spatial.rho_upper = min(1/lambda_max, 0.99);
ParamMgr.spatial.lambda_W = lambda_W;
end

function is_valid = validate_parameters(theta, ParamMgr)
theta = theta(:); 
[lb, ub] = ParamMgr.methods.get_bounds(ParamMgr);
is_valid = all(theta >= lb) && all(theta <= ub) && all(isfinite(theta));
end
