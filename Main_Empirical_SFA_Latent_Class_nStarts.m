%% Main_Empirical_SFA_Latent_Class_nStarts.m
% ============================================================================
% Spatial SFA with Latent Classes — Empirical Estimation (Table 5)
% ============================================================================
%
% Estimates the spatial SFA latent class model for 6 configurations:
%   Spec=1 (per-capita):  Full Sample (n~121), High Income (n~80), Developing (n~71)
%   Spec=2 (per-worker):  Full Sample (n~121), High Income (n~80), Developing (n~71)
%
% Corresponds to Table 5 in Du, Prokhorov & Tran (2026).
%
% DEPENDENCIES:
%   SFA_Core.m        — estimation engine (EM algorithm)
%   Empirical_Data.mat — pre-processed data (run Prepare_Data.m first)
%
% Author: Kai Du (kdu@uow.edu.au)
% ============================================================================

clear; close all; clc;

fprintf('=================================================================\n');
fprintf('   SPATIAL SFA LATENT CLASS — TABLE 5 REPLICATION\n');
fprintf('   Date: %s\n', datestr(now));
fprintf('=================================================================\n');

%% 1. LOAD DATA
data_file = 'Empirical_Data.mat';
K = 2;

loaded       = load(data_file, 'all_data', 'model_labels');
all_data     = loaded.all_data;
model_labels = loaded.model_labels;
n_models     = length(all_data);  

%% 2. ESTIMATION LOOP
all_results  = cell(n_models, 1);
all_ParamMgr = cell(n_models, 1);

for m = 1:n_models
    fprintf('========================================\n');
    fprintf('%s\n', model_labels{m});
    fprintf('========================================\n');

    data   = all_data{m};
    n_beta = size(data.X, 3);

    ParamMgr = SFA_Core.initialize_parameter_manager(K, n_beta);
    ParamMgr = SFA_Core.compute_spatial_bounds(ParamMgr, data.lambda_W);
    all_ParamMgr{m} = ParamMgr;

    init_params = SFA_Core.initialise_from_moments(data, K);

    em_tic = tic;
    [em_est, em_conv, em_omega, em_se] = SFA_Core.estimate_em(data, K, init_params, ParamMgr);
    em_time = toc(em_tic);

    em_loglik = SFA_Core.compute_loglik(em_est, data, K, ParamMgr);
    [em_est, em_se, em_omega] = resolve_label_switching(em_est, em_se, em_omega, ParamMgr);

    results      = struct();
    results.em   = struct('estimates', em_est, 'se', em_se, 'omega', em_omega, ...
                          'converged', em_conv, 'loglik', em_loglik, 'time', em_time);
    results.K    = K;
    all_results{m} = results;

    fprintf('  EM: LogLik=%.4f | Conv=%d | Time=%.0fs\n\n', em_loglik, em_conv, em_time);
end

fprintf('\n');
display_table5(all_results, model_labels, all_ParamMgr, all_data);

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function [est, se, omega] = resolve_label_switching(est, se, omega, ParamMgr)
n_beta     = ParamMgr.config.n_beta;
n_per      = ParamMgr.config.n_params_per_class;
lambda_1   = est(n_beta + 2);
lambda_2   = est(n_per  + n_beta + 2);

if lambda_1 > lambda_2
    c1 = 1:n_per;  c2 = (n_per+1):(2*n_per);
    est_new = est; est_new(c1) = est(c2); est_new(c2) = est(c1);
    se_new  = se;  se_new(c1)  = se(c2);  se_new(c2)  = se(c1);
    d_idx   = (2*n_per+1):length(est);
    est_new(d_idx) = -est(d_idx);
    est = est_new;  se = se_new;
    if ~isempty(omega) && size(omega,2) >= 2
        omega(:,[1,2]) = omega(:,[2,1]);
    end
end
end


function display_table5(results_cell, model_labels, ParamMgr_cell, data_cell)

pnames = ParamMgr_cell{1}.config.param_names;
n_p    = length(pnames);

for spec_block = 1:2
    ids = (spec_block - 1) * 3 + (1:3); 
    spec_label = model_labels{ids(1)};
    spec_label = regexp(spec_label, 'Spec=\d \([^)]+\)', 'match', 'once');

    fprintf('\n');
    fprintf('=========================================================================\n');
    fprintf('  PARAMETER ESTIMATES (EM) — %s\n', spec_label);
    fprintf('=========================================================================\n');

    % Header
    fprintf('%-16s|', '');
    for m = ids
        lbl = model_labels{m};
        lbl = regexprep(lbl, 'Spec=\d \([^)]+\), ', '');
        fprintf('  %-17s|', lbl);
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 16 + 21*3 + 1));

    for p = 1:n_p
        pname     = pnames{p};
        is_lambda = strncmp(pname, 'lambda_',   7);
        is_sigma  = strncmp(pname, 'sigma2_v_', 9);
        is_log    = is_lambda || is_sigma;

        fprintf('%-16s|', pname);
        for m = ids
            res     = results_cell{m};
            est_raw = res.em.estimates(p);
            if is_log
                fprintf('  %14.4f   |', exp(est_raw));
            else
                fprintf('  %14.4f   |', est_raw);
            end
        end
        fprintf('\n');

        fprintf('%-16s|', '');
        for m = ids
            res = results_cell{m};
            sv  = res.em.se(p);
            if isnan(sv)
                fprintf('  %14s   |', '(---)');
            else
                if is_log
                    sv = exp(res.em.estimates(p)) * sv;
                end
                fprintf('        (%8.4f)   |', sv);
            end
        end
        fprintf('\n');
    end

    fprintf('%s\n', repmat('-', 1, 16 + 21*3 + 1));

    fprintf('%-16s|', 'Log-Likelihood');
    for m = ids
        fprintf('  %14.4f   |', results_cell{m}.em.loglik);
    end
    fprintf('\n');

    fprintf('%-16s|', 'No. Countries');
    for m = ids
        fprintf('  %14d   |', data_cell{m}.n);
    end
    fprintf('\n');

end
end
