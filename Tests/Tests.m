clear; 
clc; 
close all;


fprintf('VALIDACION DEL CONTROL DLS CON RED NEURONAL\n');

rng(42);

deg = pi/180;

lambda_max  = 0.12;
dt          = 0.005;
dq_max      = 180 * deg;
alpha_gain  = 10.0;
steps_seg   = 400;
v_cmd_max   = 0.25;

eye3 = eye(3);
eye6 = eye(6);

q_min = [-360, -360, -155, -360, -360, -360]' * deg;
q_max = [ 360,  360,  155,  360,  360,  360]' * deg;

k_null = 0.8;
w_threshold_ref = 0.022;

N_samples = 60000;
n_features = 25;
lambda_grid = linspace(0, lambda_max, 101);


L(1) = Link('modified','d',0,     'a',0,    'alpha',0,     'offset',0);
L(2) = Link('modified','d',0.128, 'a',0,    'alpha',pi/2,  'offset',pi/2);
L(3) = Link('modified','d',0.1165,'a',0.274,'alpha',pi,    'offset',0);
L(4) = Link('modified','d',0.116, 'a',0.230,'alpha',pi,    'offset',-pi/2);
L(5) = Link('modified','d',0.116, 'a',0,    'alpha',-pi/2, 'offset',0);
L(6) = Link('modified','d',0.105, 'a',0,    'alpha',pi/2,  'offset',0);

bot = SerialLink(L, 'name','Dobot-CR3');

fprintf('Robot creado: %s\n', bot.name);
fprintf('Grados de libertad: %d\n\n', bot.n);

%TEST 1: MODELO CINEMÁTICO Y JACOBIANO

fprintf('TEST 1 - Modelo cinematico y Jacobiano\n');

N_model_check = 500;
model_fail = 0;
sv_min_global = inf;
cond_max_global = 0;
w_min_global = inf;
w_max_global = 0;

for i = 1:N_model_check
    q = q_min + (q_max - q_min) .* rand(6,1);

    try
        T = bot.fkine(q');
        p = T.t;

        J_full = bot.jacob0(q');
        J = J_full(1:3,:);

        [~,S,~] = svd(J,'econ');
        sv = diag(S);
        w = prod(sv);

        if numel(sv) ~= 3 || any(~isfinite(sv)) || any(~isfinite(p)) || any(size(J) ~= [3 6])
            model_fail = model_fail + 1;
            continue;
        end

        sv_min_global = min(sv_min_global, sv(end));
        cond_max_global = max(cond_max_global, sv(1)/(sv(end)+1e-12));
        w_min_global = min(w_min_global, w);
        w_max_global = max(w_max_global, w);

    catch
        model_fail = model_fail + 1;
    end
end

fprintf('Muestras revisadas                         : %d\n', N_model_check);
fprintf('Fallos numericos                           : %d\n', model_fail);
fprintf('Valor singular minimo observado            : %.8e\n', sv_min_global);
fprintf('Condicionamiento maximo observado          : %.8e\n', cond_max_global);
fprintf('Manipulabilidad minima observada           : %.8e\n', w_min_global);
fprintf('Manipulabilidad maxima observada           : %.8e\n', w_max_global);

if model_fail == 0
    fprintf('Resultado TEST 1                           : APROBADO\n\n');
else
    fprintf('Resultado TEST 1                           : REVISAR\n\n');
end

%TEST 2: GENERACIÓN DE DATOS

fprintf('TEST 2 - Generacion del conjunto de datos\n');

X_all = zeros(N_samples, n_features);
Y_all = zeros(N_samples, 1);

idx = 0;
attempts = 0;
max_attempts = N_samples * 6;

data_timer = tic;

while idx < N_samples && attempts < max_attempts
    attempts = attempts + 1;

    q = q_min + (q_max - q_min) .* rand(6,1);

    J_full = bot.jacob0(q');
    J      = J_full(1:3,:);

    [U,S,~] = svd(J,'econ');
    sv = diag(S);

    if numel(sv) < 3
        continue;
    end

    if any(~isfinite(sv)) || sv(1) < 1e-10
        continue;
    end

    w = prod(sv);
    if ~isfinite(w)
        continue;
    end

    if rand < 0.08
        v_ff = zeros(3,1);
    else
        dir_v = randn(3,1);
        dir_v = dir_v / (norm(dir_v) + 1e-12);
        v_ff  = dir_v * v_cmd_max * rand();
    end

    if rand < 0.08
        e_p = zeros(3,1);
    else
        dir_e = randn(3,1);
        dir_e = dir_e / (norm(dir_e) + 1e-12);
        e_mag = 0.05 * rand();
        e_p   = dir_e * e_mag;
    end

    gain_train = alpha_gain * (1 + 2 * min(norm(e_p), 0.05) / 0.05);
    v_cmd = v_ff + gain_train * e_p;

    nv = norm(v_cmd);
    if nv > v_cmd_max
        v_cmd = v_cmd * (v_cmd_max / nv);
    end

    lambda_star = teacher_optimal_lambda_independent( ...
        J, U, sv, q, v_cmd, norm(e_p), ...
        q_min, q_max, lambda_grid, dq_max, dt, ...
        eye3, eye6, lambda_max, k_null, v_cmd_max);

    feat = make_nn_features( ...
        q, J, U, sv, w, v_cmd, e_p, ...
        q_min, q_max, v_cmd_max);

    if any(~isfinite(feat)) || ~isfinite(lambda_star)
        continue;
    end

    idx = idx + 1;
    X_all(idx,:) = feat;
    Y_all(idx)   = lambda_star;
end

data_time = toc(data_timer);

X_all = X_all(1:idx,:);
Y_all = Y_all(1:idx);

lambda_zero_count = sum(Y_all < 0.001);
lambda_active_count = sum(Y_all >= 0.001);

fprintf('Muestras generadas                         : %d\n', idx);
fprintf('Intentos realizados                        : %d\n', attempts);
fprintf('Tiempo generacion datos                    : %.3f s\n', data_time);
fprintf('Etiquetas lambda < 0.001                   : %d (%.2f %%)\n', ...
    lambda_zero_count, 100*lambda_zero_count/max(idx,1));
fprintf('Etiquetas lambda >= 0.001                  : %d (%.2f %%)\n', ...
    lambda_active_count, 100*lambda_active_count/max(idx,1));
fprintf('Lambda minimo                              : %.8f\n', min(Y_all));
fprintf('Lambda medio                               : %.8f\n', mean(Y_all));
fprintf('Lambda maximo                              : %.8f\n', max(Y_all));

data_ok = idx == N_samples && ...
          all(isfinite(X_all(:))) && ...
          all(isfinite(Y_all(:))) && ...
          min(Y_all) >= -1e-12 && ...
          max(Y_all) <= lambda_max + 1e-12 && ...
          lambda_zero_count > 0 && ...
          lambda_active_count > 0;

if data_ok
    fprintf('Resultado TEST 2                           : APROBADO\n\n');
else
    fprintf('Resultado TEST 2                           : REVISAR\n\n');
end

%TEST 3: DIVISIÓN Y NORMALIZACIÓN


fprintf('TEST 3 - Division train/validacion/test y normalizacion\n');

perm = randperm(idx);
X_all = X_all(perm,:);
Y_all = Y_all(perm);

n_train = round(0.70 * idx);
n_val   = round(0.15 * idx);
n_test  = idx - n_train - n_val;

X_raw_tr   = X_all(1:n_train,:);
Y_raw_tr   = Y_all(1:n_train);

X_raw_val  = X_all(n_train+1:n_train+n_val,:);
Y_raw_val  = Y_all(n_train+1:n_train+n_val);

X_raw_test = X_all(n_train+n_val+1:end,:);
Y_raw_test = Y_all(n_train+n_val+1:end);

mu_X  = mean(X_raw_tr, 1);
sig_X = std(X_raw_tr, 0, 1) + 1e-8;

X_tr   = ((X_raw_tr   - mu_X) ./ sig_X)';
X_val  = ((X_raw_val  - mu_X) ./ sig_X)';
X_test = ((X_raw_test - mu_X) ./ sig_X)';

Y_tr   = (Y_raw_tr   / lambda_max)';
Y_val  = (Y_raw_val  / lambda_max)';
Y_test = (Y_raw_test / lambda_max)';

train_mean_max = max(abs(mean(X_tr,2)));
train_std_error = max(abs(std(X_tr,0,2) - 1));

fprintf('Train                                      : %d\n', n_train);
fprintf('Validacion                                : %d\n', n_val);
fprintf('Test                                       : %d\n', n_test);
fprintf('Max |media train normalizada|              : %.8e\n', train_mean_max);
fprintf('Max |std train normalizada - 1|            : %.8e\n', train_std_error);
fprintf('Rango Y_train normalizado                  : [%.6f, %.6f]\n', min(Y_tr), max(Y_tr));
fprintf('Rango Y_val normalizado                    : [%.6f, %.6f]\n', min(Y_val), max(Y_val));
fprintf('Rango Y_test normalizado                   : [%.6f, %.6f]\n', min(Y_test), max(Y_test));

norm_ok = train_mean_max < 1e-8 && ...
          train_std_error < 1e-4 && ...
          all(Y_tr >= -1e-12) && all(Y_tr <= 1 + 1e-12) && ...
          all(Y_val >= -1e-12) && all(Y_val <= 1 + 1e-12) && ...
          all(Y_test >= -1e-12) && all(Y_test <= 1 + 1e-12);

if norm_ok
    fprintf('Resultado TEST 3                           : APROBADO\n\n');
else
    fprintf('Resultado TEST 3                           : REVISAR\n\n');
end

%TEST 4: ENTRENAMIENTO DE LA RED

fprintf('TEST 4 - Entrenamiento de la red neuronal\n');

arch = [n_features, 128, 256, 128, 64, 1];
n_layers = length(arch) - 1;

W = cell(n_layers,1);
b = cell(n_layers,1);

for l = 1:n_layers
    fan_in  = arch(l);
    fan_out = arch(l+1);
    W{l} = randn(fan_out, fan_in) * sqrt(2 / fan_in);
    b{l} = zeros(fan_out, 1);
end

lr         = 1e-3;
lr_decay   = 0.92;
n_epochs   = 200;
batch_size = 512;

beta1      = 0.9;
beta2      = 0.999;
eps_adam   = 1e-8;

mW = cell(n_layers,1); vW = cell(n_layers,1);
mb = cell(n_layers,1); vb = cell(n_layers,1);

for l = 1:n_layers
    mW{l} = zeros(size(W{l}));
    vW{l} = zeros(size(W{l}));
    mb{l} = zeros(size(b{l}));
    vb{l} = zeros(size(b{l}));
end

loss_hist = zeros(n_epochs, 1);
val_hist  = zeros(n_epochs, 1);

t_adam     = 0;
best_val   = inf;
best_W     = W;
best_b     = b;
patience   = 25;
no_improve = 0;
ep_done    = 0;

train_timer = tic;

for ep = 1:n_epochs
    ep_done = ep;

    perm_ep = randperm(n_train);
    X_sh = X_tr(:, perm_ep);
    Y_sh = Y_tr(perm_ep);

    n_batches = ceil(n_train / batch_size);
    ep_loss = 0;
    used_batches = 0;

    for b_idx = 1:n_batches
        i1 = (b_idx-1)*batch_size + 1;
        i2 = min(b_idx*batch_size, n_train);
        bs = i2 - i1 + 1;

        Xb = X_sh(:, i1:i2);
        Yb = Y_sh(i1:i2);

        A_cache = cell(n_layers+1, 1);
        Z_cache = cell(n_layers, 1);
        A_cache{1} = Xb;

        for l = 1:n_layers-1
            Z_cache{l}   = W{l} * A_cache{l} + b{l};
            A_cache{l+1} = max(0, Z_cache{l});
        end

        Z_cache{n_layers} = W{n_layers} * A_cache{n_layers} + b{n_layers};
        Z_out = max(min(Z_cache{n_layers}, 60), -60);
        A_cache{n_layers+1} = 1 ./ (1 + exp(-Z_out));

        Y_hat = A_cache{n_layers+1};

        diff = Y_hat - Yb;
        loss = mean(diff.^2);

        ep_loss = ep_loss + loss;
        used_batches = used_batches + 1;

        sig_out = A_cache{n_layers+1};
        dZ_out  = (2 / bs) * diff .* sig_out .* (1 - sig_out);

        dW_acc = cell(n_layers, 1);
        dB_acc = cell(n_layers, 1);

        dW_acc{n_layers} = dZ_out * A_cache{n_layers}';
        dB_acc{n_layers} = sum(dZ_out, 2);

        dA_prev = W{n_layers}' * dZ_out;

        for l = n_layers-1:-1:1
            dZ = dA_prev .* (Z_cache{l} > 0);

            dW_acc{l} = dZ * A_cache{l}';
            dB_acc{l} = sum(dZ, 2);

            if l > 1
                dA_prev = W{l}' * dZ;
            end
        end

        t_adam = t_adam + 1;
        bc1 = 1 - beta1^t_adam;
        bc2 = 1 - beta2^t_adam;

        for l = 1:n_layers
            mW{l} = beta1*mW{l} + (1-beta1)*dW_acc{l};
            vW{l} = beta2*vW{l} + (1-beta2)*dW_acc{l}.^2;

            mb{l} = beta1*mb{l} + (1-beta1)*dB_acc{l};
            vb{l} = beta2*vb{l} + (1-beta2)*dB_acc{l}.^2;

            W{l} = W{l} - lr * (mW{l}/bc1) ./ (sqrt(vW{l}/bc2) + eps_adam);
            b{l} = b{l} - lr * (mb{l}/bc1) ./ (sqrt(vb{l}/bc2) + eps_adam);
        end
    end

    Y_val_hat = nn_forward(X_val, W, b, n_layers);
    val_loss  = mean((Y_val_hat - Y_val).^2);

    loss_hist(ep) = ep_loss / used_batches;
    val_hist(ep)  = val_loss;

    if val_loss < best_val
        best_val   = val_loss;
        best_W     = W;
        best_b     = b;
        no_improve = 0;
    else
        no_improve = no_improve + 1;
    end

    if mod(ep, 15) == 0
        lr = lr * lr_decay;
    end

    if mod(ep,10)==0 || ep==1
        val_mae_lam = mean(abs(Y_val_hat - Y_val)) * lambda_max;
        fprintf('Epoca %3d/%d | Loss %.8f | Val %.8f | MAE lambda %.8f | LR %.6f\n', ...
            ep, n_epochs, loss_hist(ep), val_hist(ep), val_mae_lam, lr);
    end

    if no_improve >= patience
        fprintf('Early stopping en epoca %d\n', ep);
        break;
    end
end

train_time = toc(train_timer);

W = best_W;
b = best_b;

Y_test_hat = nn_forward(X_test, W, b, n_layers);
test_mse = mean((Y_test_hat - Y_test).^2);
test_mae_lambda = mean(abs(Y_test_hat - Y_test)) * lambda_max;

Y_const_hat = mean(Y_tr) * ones(size(Y_test));
const_mse = mean((Y_const_hat - Y_test).^2);
const_mae_lambda = mean(abs(Y_const_hat - Y_test)) * lambda_max;

mae_reduction = 100 * (const_mae_lambda - test_mae_lambda) / ...
                (abs(const_mae_lambda) + 1e-12);

fprintf('\nTiempo entrenamiento                       : %.3f s\n', train_time);
fprintf('Epocas ejecutadas                          : %d\n', ep_done);
fprintf('Mejor Val MSE                              : %.8f\n', best_val);
fprintf('Test MSE red                               : %.8f\n', test_mse);
fprintf('Test MAE lambda red                        : %.8f\n', test_mae_lambda);
fprintf('Test MSE predictor constante               : %.8f\n', const_mse);
fprintf('Test MAE lambda predictor constante        : %.8f\n', const_mae_lambda);
fprintf('Reduccion MAE contra predictor constante   : %.2f %%\n', mae_reduction);

train_ok = all(isfinite(Y_test_hat)) && ...
           all(Y_test_hat >= -1e-12) && ...
           all(Y_test_hat <= 1 + 1e-12) && ...
           test_mae_lambda < const_mae_lambda;

if train_ok
    fprintf('Resultado TEST 4                           : APROBADO\n\n');
else
    fprintf('Resultado TEST 4                           : REVISAR\n\n');
end

net.W          = W;
net.b          = b;
net.n_layers   = n_layers;
net.mu_X       = mu_X;
net.sig_X      = sig_X;
net.beta_lam   = 0.60;
net.lambda_max = lambda_max;

params.lambda_max      = lambda_max;
params.dt              = dt;
params.dq_max          = dq_max;
params.alpha_gain      = alpha_gain;
params.k_null          = k_null;
params.v_cmd_max       = v_cmd_max;
params.q_min           = q_min;
params.q_max           = q_max;
params.eye3            = eye3;
params.eye6            = eye6;
params.w_threshold_ref = w_threshold_ref;
params.lambda_const    = mean(Y_raw_tr);

%TEST 5: RESPUESTA DE LAMBDA EN ESTADOS NUEVOS

fprintf('TEST 5 - Respuesta de lambda en estados nuevos\n');

N_probe = 2000;

lambda_nn_probe = zeros(N_probe,1);
lambda_yosh_probe = zeros(N_probe,1);
w_probe = zeros(N_probe,1);
svmin_probe = zeros(N_probe,1);
cmdnorm_probe = zeros(N_probe,1);

for i = 1:N_probe
    q = q_min + (q_max - q_min) .* rand(6,1);

    J_full = bot.jacob0(q');
    J = J_full(1:3,:);

    [U,S,~] = svd(J,'econ');
    sv = diag(S);
    w = prod(sv);

    if rand < 0.08
        v_ff = zeros(3,1);
    else
        dir_v = randn(3,1);
        dir_v = dir_v / (norm(dir_v) + 1e-12);
        v_ff  = dir_v * v_cmd_max * rand();
    end

    if rand < 0.08
        e_p = zeros(3,1);
    else
        dir_e = randn(3,1);
        dir_e = dir_e / (norm(dir_e) + 1e-12);
        e_mag = 0.05 * rand();
        e_p   = dir_e * e_mag;
    end

    gain_i = alpha_gain * (1 + 2 * min(norm(e_p), 0.05) / 0.05);
    v_cmd = v_ff + gain_i * e_p;

    nv = norm(v_cmd);
    if nv > v_cmd_max
        v_cmd = v_cmd * (v_cmd_max / nv);
    end

    feat = make_nn_features(q, J, U, sv, w, v_cmd, e_p, q_min, q_max, v_cmd_max);
    feat_norm = ((feat - mu_X) ./ sig_X)';

    y_nn = nn_forward(feat_norm, W, b, n_layers);
    lambda_nn_probe(i) = max(0, min(1, y_nn)) * lambda_max;

    lambda_yosh_probe(i) = yoshikawa_base_lambda(w, w_threshold_ref, lambda_max);

    w_probe(i) = w;
    svmin_probe(i) = sv(end);
    cmdnorm_probe(i) = norm(v_cmd);
end

lambda_std = std(lambda_nn_probe);
lambda_range = max(lambda_nn_probe) - min(lambda_nn_probe);
lambda_active_probe = mean(lambda_nn_probe > 0.001) * 100;
lambda_diff_mean = mean(abs(lambda_nn_probe - lambda_yosh_probe));
lambda_diff_rate = mean(abs(lambda_nn_probe - lambda_yosh_probe) > 1e-4) * 100;

corr_probe = corrcoef(lambda_nn_probe, lambda_yosh_probe);
if all(isfinite(corr_probe(:))) && size(corr_probe,1) == 2
    corr_nn_yosh = corr_probe(1,2);
else
    corr_nn_yosh = NaN;
end

fprintf('Muestras nuevas evaluadas                  : %d\n', N_probe);
fprintf('Lambda NN minimo                           : %.8f\n', min(lambda_nn_probe));
fprintf('Lambda NN maximo                           : %.8f\n', max(lambda_nn_probe));
fprintf('Lambda NN desviacion estandar              : %.8f\n', lambda_std);
fprintf('Lambda NN rango                            : %.8f\n', lambda_range);
fprintf('Lambda NN activo > 0.001                   : %.2f %%\n', lambda_active_probe);
fprintf('Diferencia media |lambda_NN - lambda_Yosh| : %.8f\n', lambda_diff_mean);
fprintf('Porcentaje con diferencia > 1e-4           : %.2f %%\n', lambda_diff_rate);
fprintf('Correlacion lambda_NN vs lambda_Yosh       : %.6f\n', corr_nn_yosh);

probe_ok = all(isfinite(lambda_nn_probe)) && ...
           min(lambda_nn_probe) >= -1e-12 && ...
           max(lambda_nn_probe) <= lambda_max + 1e-12 && ...
           lambda_std > 1e-8 && ...
           lambda_diff_rate > 0;

if probe_ok
    fprintf('Resultado TEST 5                           : APROBADO\n\n');
else
    fprintf('Resultado TEST 5                           : REVISAR\n\n');
end

%TEST 6: SIMULACIÓN DE TRAYECTORIA BASE

fprintf('TEST 6 - Simulacion de trayectoria base\n');

altura = 0.20;
use_stress_test = false;

[P_all_base, v_ff_base, q0_base] = prepare_trajectory_case( ...
    bot, steps_seg, use_stress_test, altura, dt, v_cmd_max, q_min, q_max);

N_base = size(P_all_base,1);

res_base_zero  = simulate_controller_validation(bot, P_all_base, altura, v_ff_base, q0_base, params, net, 'ZERO');
res_base_const = simulate_controller_validation(bot, P_all_base, altura, v_ff_base, q0_base, params, net, 'CONST');
res_base_yosh  = simulate_controller_validation(bot, P_all_base, altura, v_ff_base, q0_base, params, net, 'YOSHREF');
res_base_nn    = simulate_controller_validation(bot, P_all_base, altura, v_ff_base, q0_base, params, net, 'NN');

print_metrics('BASE - ZERO', res_base_zero, N_base);
print_metrics('BASE - CONST', res_base_const, N_base);
print_metrics('BASE - YOSHREF', res_base_yosh, N_base);
print_metrics('BASE - NN', res_base_nn, N_base);

base_ok = result_is_finite(res_base_nn) && ...
          res_base_nn.metrics.q_viol_max <= 1e-10 && ...
          res_base_nn.metrics.sat_steps == 0;

if base_ok
    fprintf('\nResultado TEST 6                           : APROBADO\n\n');
else
    fprintf('\nResultado TEST 6                           : REVISAR\n\n');
end

%TEST 7: SIMULACIÓN CON PUNTO DE ESTRS

fprintf('TEST 7 - Simulacion con punto de estres\n');

use_stress_test = true;

[P_all_stress, v_ff_stress, q0_stress] = prepare_trajectory_case( ...
    bot, steps_seg, use_stress_test, altura, dt, v_cmd_max, q_min, q_max);

N_stress = size(P_all_stress,1);

res_stress_zero  = simulate_controller_validation(bot, P_all_stress, altura, v_ff_stress, q0_stress, params, net, 'ZERO');
res_stress_const = simulate_controller_validation(bot, P_all_stress, altura, v_ff_stress, q0_stress, params, net, 'CONST');
res_stress_yosh  = simulate_controller_validation(bot, P_all_stress, altura, v_ff_stress, q0_stress, params, net, 'YOSHREF');
res_stress_nn    = simulate_controller_validation(bot, P_all_stress, altura, v_ff_stress, q0_stress, params, net, 'NN');

print_metrics('ESTRES - ZERO', res_stress_zero, N_stress);
print_metrics('ESTRES - CONST', res_stress_const, N_stress);
print_metrics('ESTRES - YOSHREF', res_stress_yosh, N_stress);
print_metrics('ESTRES - NN', res_stress_nn, N_stress);

stress_ok = result_is_finite(res_stress_nn) && ...
            res_stress_nn.metrics.q_viol_max <= 1e-10;

if stress_ok
    fprintf('\nResultado TEST 7                           : APROBADO\n\n');
else
    fprintf('\nResultado TEST 7                           : REVISAR\n\n');
end


fprintf('RESUMEN DE VALIDACIONES\n');
test_names = { ...
    'Modelo cinematico y Jacobiano'; ...
    'Generacion de datos'; ...
    'Division y normalizacion'; ...
    'Entrenamiento y test'; ...
    'Respuesta lambda en estados nuevos'; ...
    'Trayectoria base'; ...
    'Trayectoria con punto de estres'};

test_status = { ...
    status_text(model_fail == 0); ...
    status_text(data_ok); ...
    status_text(norm_ok); ...
    status_text(train_ok); ...
    status_text(probe_ok); ...
    status_text(base_ok); ...
    status_text(stress_ok)};

T_tests = table(test_names, test_status, ...
    'VariableNames', {'Prueba','Resultado'});

disp(T_tests);

metric_rows = {};
metric_rows = [metric_rows; metrics_row('BASE','ZERO',res_base_zero)];
metric_rows = [metric_rows; metrics_row('BASE','CONST',res_base_const)];
metric_rows = [metric_rows; metrics_row('BASE','YOSHREF',res_base_yosh)];
metric_rows = [metric_rows; metrics_row('BASE','NN',res_base_nn)];
metric_rows = [metric_rows; metrics_row('ESTRES','ZERO',res_stress_zero)];
metric_rows = [metric_rows; metrics_row('ESTRES','CONST',res_stress_const)];
metric_rows = [metric_rows; metrics_row('ESTRES','YOSHREF',res_stress_yosh)];
metric_rows = [metric_rows; metrics_row('ESTRES','NN',res_stress_nn)];

T_metrics = cell2table(metric_rows, ...
    'VariableNames', { ...
    'Caso','Controlador','Score','ErrorMedio_m','ErrorMaximo_m', ...
    'DqMax_rad_s','DqNormaMedia_rad_s','PasosSaturacion', ...
    'ArticulacionesSaturadas','ViolacionLimite_rad', ...
    'GananciaInvMedia','GananciaInvMaxima','LambdaMedia', ...
    'LambdaMaxima','LambdaActiva_pct','Tiempo_s'});

disp(T_metrics);



%FUNCIONES LOCALES/AUX

function [P_all, v_ff_all, q0] = prepare_trajectory_case( ...
    bot, steps_seg, use_stress_test, altura, dt, v_cmd_max, q_min, q_max)

    P_all = build_letter_p_trajectory(steps_seg, use_stress_test);
    N_total = size(P_all,1);

    P3_all = [P_all, altura*ones(N_total,1)];

    v_ff_all = zeros(3, N_total);
    v_ff_all(:,2:end) = (P3_all(2:end,:) - P3_all(1:end-1,:))' / dt;

    v_norm_all = sqrt(sum(v_ff_all.^2, 1));
    scale_v = min(1, v_cmd_max ./ (v_norm_all + 1e-12));
    v_ff_all = v_ff_all .* scale_v;

    pt = @(x,y,z) transl(x,y,z) * trotx(0);
    maskXYZ = [1 1 1 0 0 0];

    T0 = pt(P_all(1,1), P_all(1,2), altura);

    try
        q0 = bot.ikine(T0, 'mask', maskXYZ);
    catch
        q0 = [];
    end

    if isempty(q0) || any(~isfinite(q0))
        q0 = [0, 0, 0, 0, 0, 0];
    end

    q0 = q0(:);
    q0 = max(min(q0, q_max), q_min);
end

% ----------------------------------------------------------------
function P_all = build_letter_p_trajectory(steps_seg, use_stress_test)

    A=[0.15,0.15];  B=[0.15,0.40];  C=[0.175,0.425];
    D=[0.475,0.425]; F=[0.505,0.275]; G=[0.505,0.15];
    H=[0.41,0.15];  I=[0.335,0.15];
    Jp=[0.26,0.15]; K=[0.26,0.24];

    mk = @(p1,p2,n) [linspace(p1(1),p2(1),n)', ...
                     linspace(p1(2),p2(2),n)'];

    segs = {};

    segs{end+1} = mk(A,B,steps_seg);
    segs{end+1} = mk(B,C,steps_seg);
    segs{end+1} = mk(C,D,steps_seg);

    centroDer = [0.5049936661454, 0.3529987332291];
    radioDer  = sqrt(0.0060838024255);

    angD = atan2(D(2)-centroDer(2), D(1)-centroDer(1));
    angF = atan2(F(2)-centroDer(2), F(1)-centroDer(1));

    if angF < angD
        angF = angF + 2*pi;
    end

    angArcDer = linspace(angD, angF, steps_seg);

    arc_der = [centroDer(1)+radioDer*cos(angArcDer(:)), ...
               centroDer(2)+radioDer*sin(angArcDer(:))];

    segs{end+1} = arc_der;
    segs{end+1} = mk(F,G,steps_seg);
    segs{end+1} = mk(G,H,steps_seg);

    centro_inf = 0.5 * (H + Jp);
    radio_inf  = norm(H - Jp) / 2;

    ang_H  = atan2(H(2)  - centro_inf(2), H(1)  - centro_inf(1));
    ang_Jp = atan2(Jp(2) - centro_inf(2), Jp(1) - centro_inf(1));

    if ang_Jp > ang_H
        ang_Jp = ang_Jp - 2*pi;
    end

    angArcInf = linspace(ang_H, ang_Jp, steps_seg);

    arc_inf = [centro_inf(1)+radio_inf*cos(angArcInf(:)), ...
               centro_inf(2)+radio_inf*sin(angArcInf(:))];

    segs{end+1} = arc_inf;
    segs{end+1} = mk(Jp,K,steps_seg);
    segs{end+1} = mk(K,A,steps_seg);

    total_pts = size(segs{1},1);
    for s = 2:length(segs)
        total_pts = total_pts + size(segs{s},1) - 1;
    end

    P_all = zeros(total_pts,2);
    posP = 1;

    P_all(posP:posP+size(segs{1},1)-1,:) = segs{1};
    posP = posP + size(segs{1},1);

    for s = 2:length(segs)
        nseg = size(segs{s},1) - 1;
        P_all(posP:posP+nseg-1,:) = segs{s}(2:end,:);
        posP = posP + nseg;
    end

    if use_stress_test
        idx_sing = round(size(P_all,1)*0.48);
        P_sing   = [0.530, 0.265];
        n_bridge = round(steps_seg/2);

        P_before = P_all(idx_sing,:);
        P_after  = P_all(idx_sing+1,:);

        bridge1  = mk(P_before, P_sing, n_bridge);
        bridge2  = mk(P_sing, P_after,  n_bridge);

        P_all = [P_all(1:idx_sing-1,:); ...
                 bridge1; ...
                 bridge2(2:end,:); ...
                 P_all(idx_sing+2:end,:)];
    end
end

% ----------------------------------------------------------------
function res = simulate_controller_validation(bot, P_all, altura, v_ff_all, q0, params, net, mode)

    N_total = size(P_all,1);
    q = q0(:);

    q_hist            = zeros(N_total, 6);
    w_hist            = zeros(1, N_total);
    lambda_hist       = zeros(1, N_total);
    lambda_raw_hist   = zeros(1, N_total);
    lambda_yosh_hist  = zeros(1, N_total);
    pos_hist          = zeros(3, N_total);
    err_hist          = zeros(1, N_total);
    dq_abs_max_hist   = zeros(1, N_total);
    dq_norm_hist      = zeros(1, N_total);
    dq_sat_hist       = zeros(1, N_total);
    v_abs_err_hist    = zeros(1, N_total);
    v_rel_err_hist    = NaN(1, N_total);
    v_cmd_norm_hist   = zeros(1, N_total);
    q_limit_viol      = zeros(1, N_total);
    inv_gain_max_hist = zeros(1, N_total);
    sing_index_hist   = zeros(1, N_total);

    v_cmd_eps = 1e-4;
    lam_prev = 0;

    sim_timer = tic;

    for k = 1:N_total

        p_d = [P_all(k,1); P_all(k,2); altura];

        T_cur = bot.fkine(q');
        p_cur = T_cur.t;

        e_p = p_d - p_cur;

        J_full = bot.jacob0(q');
        J      = J_full(1:3,:);

        [U,S,~] = svd(J,'econ');
        sv = diag(S);
        w  = prod(sv);

        gain_k = params.alpha_gain * (1 + 2 * min(norm(e_p), 0.05) / 0.05);
        v_cmd = v_ff_all(:,k) + gain_k * e_p;

        nv = norm(v_cmd);
        if nv > params.v_cmd_max
            v_cmd = v_cmd * (params.v_cmd_max / nv);
        end

        lambda_yosh = yoshikawa_base_lambda(w, params.w_threshold_ref, params.lambda_max);

        switch upper(mode)
            case 'NN'
                feat = make_nn_features(q, J, U, sv, w, v_cmd, e_p, ...
                                        params.q_min, params.q_max, params.v_cmd_max);

                feat_norm = ((feat - net.mu_X) ./ net.sig_X)';

                lambda_pred = nn_forward(feat_norm, net.W, net.b, net.n_layers);
                lambda_pred = max(0, min(1, lambda_pred));

                lam_raw = lambda_pred * net.lambda_max;
                lam_raw = max(0, min(params.lambda_max, lam_raw));

                lam = net.beta_lam * lam_prev + (1 - net.beta_lam) * lam_raw;
                lam = max(0, min(params.lambda_max, lam));
                lam_prev = lam;

            case 'ZERO'
                lam_raw = 0;
                lam = 0;

            case 'CONST'
                lam_raw = params.lambda_const;
                lam = params.lambda_const;

            case 'YOSHREF'
                lam_raw = lambda_yosh;
                lam = lambda_yosh;

            otherwise
                error('Modo de controlador no reconocido');
        end

        JJT = J * J';

        A_dls = JJT + (lam^2 + 1e-10) * params.eye3;
        J_dls = J' * (A_dls \ params.eye3);

        dq_task = J_dls * v_cmd;

        dq_null = joint_center_velocity(q, params.q_min, params.q_max, params.k_null);
        N_null  = params.eye6 - J_dls * J;

        dq_raw = dq_task + N_null * dq_null;

        dq_abs_max_hist(k) = max(abs(dq_raw));
        dq_norm_hist(k)    = norm(dq_raw);
        dq_sat_hist(k)     = sum(abs(dq_raw) > params.dq_max);

        dq = max(min(dq_raw, params.dq_max), -params.dq_max);

        v_real = J * dq;

        v_cmd_norm_hist(k) = norm(v_cmd);
        v_abs_err_hist(k)  = norm(v_cmd - v_real);

        if v_cmd_norm_hist(k) > v_cmd_eps
            v_rel_err_hist(k) = v_abs_err_hist(k) / v_cmd_norm_hist(k);
        end

        q = q + dq * params.dt;

        q_low_viol  = max(0, params.q_min - q);
        q_high_viol = max(0, q - params.q_max);
        q_limit_viol(k) = max([q_low_viol; q_high_viol; 0]);

        T_new = bot.fkine(q');
        p_new = T_new.t;

        q_hist(k,:)          = q';
        w_hist(k)            = w;
        lambda_hist(k)       = lam;
        lambda_raw_hist(k)   = lam_raw;
        lambda_yosh_hist(k)  = lambda_yosh;
        pos_hist(:,k)        = p_new;
        err_hist(k)          = norm(p_d - p_new);

        inv_gain = sv ./ (sv.^2 + lam^2 + 1e-12);
        inv_gain_max_hist(k) = max(inv_gain);

        sing_index_hist(k) = 1 - sv(end) / (sv(1) + 1e-12);
    end

    sim_time = toc(sim_timer);

    res.mode              = mode;
    res.q_hist            = q_hist;
    res.w_hist            = w_hist;
    res.lambda_hist       = lambda_hist;
    res.lambda_raw_hist   = lambda_raw_hist;
    res.lambda_yosh_hist  = lambda_yosh_hist;
    res.pos_hist          = pos_hist;
    res.err_hist          = err_hist;
    res.dq_abs_max_hist   = dq_abs_max_hist;
    res.dq_norm_hist      = dq_norm_hist;
    res.dq_sat_hist       = dq_sat_hist;
    res.v_abs_err_hist    = v_abs_err_hist;
    res.v_rel_err_hist    = v_rel_err_hist;
    res.v_cmd_norm_hist   = v_cmd_norm_hist;
    res.q_limit_viol      = q_limit_viol;
    res.inv_gain_max_hist = inv_gain_max_hist;
    res.sing_index_hist   = sing_index_hist;

    valid_vrel = v_rel_err_hist(isfinite(v_rel_err_hist));

    if isempty(valid_vrel)
        vrel_mean = NaN;
        vrel_max  = NaN;
        rel_score = 0;
    else
        vrel_mean = mean(valid_vrel);
        vrel_max  = max(valid_vrel);
        rel_score = mean(valid_vrel.^2);
    end

    res.metrics.err_mean       = mean(err_hist);
    res.metrics.err_max        = max(err_hist);
    res.metrics.dq_raw_max     = max(dq_abs_max_hist);
    res.metrics.dq_norm_mean   = mean(dq_norm_hist);
    res.metrics.sat_steps      = sum(dq_sat_hist > 0);
    res.metrics.sat_joints     = sum(dq_sat_hist);
    res.metrics.v_abs_mean     = mean(v_abs_err_hist);
    res.metrics.v_abs_max      = max(v_abs_err_hist);
    res.metrics.v_rel_mean     = vrel_mean;
    res.metrics.v_rel_max      = vrel_max;
    res.metrics.lambda_max     = max(lambda_hist);
    res.metrics.lambda_mean    = mean(lambda_hist);
    res.metrics.lambda_active  = sum(lambda_hist > 0.001);
    res.metrics.q_viol_max     = max(q_limit_viol);
    res.metrics.inv_gain_max   = max(inv_gain_max_hist);
    res.metrics.inv_gain_mean  = mean(inv_gain_max_hist);
    res.metrics.sim_time       = sim_time;
    res.metrics.draw_time      = 0;
    res.metrics.avg_step_time  = sim_time / N_total;

    res.metrics.score = ...
        rel_score + ...
        0.03 * mean((dq_abs_max_hist / params.dq_max).^2) + ...
        5.00 * mean(dq_sat_hist > 0) + ...
        50.0 * max(q_limit_viol)^2 + ...
        0.02 * mean(lambda_hist / params.lambda_max);
end

% ----------------------------------------------------------------
function print_metrics(name, res, N_total)

    fprintf('\n--- %s ---\n', name);
    fprintf('Score global                              : %.8f\n', res.metrics.score);
    fprintf('lambda maximo                            : %.6f\n', res.metrics.lambda_max);
    fprintf('lambda medio                             : %.6f\n', res.metrics.lambda_mean);
    fprintf('Pasos lambda > 0.001                     : %d/%d (%.1f%%)\n', ...
            res.metrics.lambda_active, N_total, 100*res.metrics.lambda_active/N_total);
    fprintf('Error cartesiano medio                   : %.6f m\n', res.metrics.err_mean);
    fprintf('Error cartesiano maximo                  : %.6f m\n', res.metrics.err_max);
    fprintf('dq maximo antes de saturar               : %.6f rad/s\n', res.metrics.dq_raw_max);
    fprintf('dq norma media                           : %.6f rad/s\n', res.metrics.dq_norm_mean);
    fprintf('Pasos con saturacion dq                  : %d/%d (%.1f%%)\n', ...
            res.metrics.sat_steps, N_total, 100*res.metrics.sat_steps/N_total);
    fprintf('Total articulaciones saturadas           : %d\n', res.metrics.sat_joints);
    fprintf('Error abs medio velocidad cart.          : %.6f m/s\n', res.metrics.v_abs_mean);
    fprintf('Error abs max velocidad cart.            : %.6f m/s\n', res.metrics.v_abs_max);

    if isfinite(res.metrics.v_rel_mean)
        fprintf('Error relativo medio velocidad cart.     : %.6f\n', res.metrics.v_rel_mean);
        fprintf('Error relativo max velocidad cart.       : %.6f\n', res.metrics.v_rel_max);
    else
        fprintf('Error relativo velocidad cart.           : no definido\n');
    end

    fprintf('Ganancia inversa efectiva media          : %.6f\n', res.metrics.inv_gain_mean);
    fprintf('Ganancia inversa efectiva maxima         : %.6f\n', res.metrics.inv_gain_max);
    fprintf('Violacion maxima limite articular        : %.8f rad\n', res.metrics.q_viol_max);
    fprintf('Tiempo simulacion/control                : %.3f s\n', res.metrics.sim_time);
    fprintf('Tiempo promedio por paso                 : %.6f s/paso\n', res.metrics.avg_step_time);
end

% ----------------------------------------------------------------
function ok = result_is_finite(res)

    ok = all(isfinite(res.q_hist(:))) && ...
         all(isfinite(res.pos_hist(:))) && ...
         all(isfinite(res.err_hist(:))) && ...
         all(isfinite(res.lambda_hist(:))) && ...
         all(isfinite(res.dq_abs_max_hist(:))) && ...
         all(isfinite(res.inv_gain_max_hist(:))) && ...
         isfinite(res.metrics.score);
end

% ----------------------------------------------------------------
function row = metrics_row(caso, controlador, res)

    row = { ...
        caso, ...
        controlador, ...
        res.metrics.score, ...
        res.metrics.err_mean, ...
        res.metrics.err_max, ...
        res.metrics.dq_raw_max, ...
        res.metrics.dq_norm_mean, ...
        res.metrics.sat_steps, ...
        res.metrics.sat_joints, ...
        res.metrics.q_viol_max, ...
        res.metrics.inv_gain_mean, ...
        res.metrics.inv_gain_max, ...
        res.metrics.lambda_mean, ...
        res.metrics.lambda_max, ...
        100 * res.metrics.lambda_active / numel(res.lambda_hist), ...
        res.metrics.sim_time ...
    };
end

% ----------------------------------------------------------------
function txt = status_text(ok)

    if ok
        txt = 'APROBADO';
    else
        txt = 'REVISAR';
    end
end

% ----------------------------------------------------------------
function lambda_star = teacher_optimal_lambda_independent( ...
    J, U, sv, q, v_cmd, e_norm, ...
    q_min, q_max, lambda_grid, dq_max, dt, ...
    eye3, eye6, lambda_max, k_null, v_cmd_max)

    JJT = J * J';
    v_norm = norm(v_cmd);
    v_norm2 = v_norm^2 + 1e-12;

    range_q = q_max - q_min;

    dq_null = joint_center_velocity(q, q_min, q_max, k_null);

    gain_safe = 0.65 * dq_max / (v_cmd_max + 1e-12);

    if v_norm < 1e-12
        dir_weight = ones(3,1) / sqrt(3);
        cmd_activity = 0;
    else
        dir_weight = abs(U' * v_cmd) / (v_norm + 1e-12);
        dir_weight = min(dir_weight, 1);
        cmd_activity = min(1, v_norm / (0.20 * v_cmd_max + 1e-12));
    end

    e_scaled = min(e_norm, 0.05) / 0.05;
    track_weight = 1 + 4 * e_scaled;

    best_cost = inf;
    lambda_star = 0;

    for i = 1:numel(lambda_grid)
        lam = lambda_grid(i);

        A = JJT + (lam^2 + 1e-10) * eye3;
        J_dls = J' * (A \ eye3);

        dq_task = J_dls * v_cmd;
        N_null  = eye6 - J_dls * J;
        dq      = dq_task + N_null * dq_null;

        dq_sat = max(min(dq, dq_max), -dq_max);
        v_real = J * dq_sat;

        tracking_cost = norm(v_cmd - v_real)^2 / v_norm2;
        joint_speed_cost = mean((dq / dq_max).^2);
        saturation_cost = mean((max(0, abs(dq) - dq_max) / dq_max).^2);

        q_next = q + dq_sat * dt;

        margin_next = min(q_next - q_min, q_max - q_next) ./ (0.5 * range_q);
        margin_next = max(-1, min(1, margin_next));

        near_limit_cost = mean(max(0, 0.12 - margin_next).^2) / (0.12^2);

        violation_low  = max(0, q_min - q_next) ./ range_q;
        violation_high = max(0, q_next - q_max) ./ range_q;
        limit_violation_cost = sum(violation_low.^2 + violation_high.^2);

        inv_gain = sv ./ (sv.^2 + lam^2 + 1e-12);

        directional_gain = norm(inv_gain .* dir_weight);
        max_inv_gain     = max(inv_gain);

        directional_gain_cost = max(0, (directional_gain - gain_safe) / (gain_safe + 1e-12))^2;
        isotropic_gain_cost   = max(0, (max_inv_gain - gain_safe) / (gain_safe + 1e-12))^2;

        regularization_cost = (lam / lambda_max)^2;

        cost = ...
            track_weight * tracking_cost + ...
            0.04 * joint_speed_cost + ...
            8.00 * saturation_cost + ...
            0.25 * near_limit_cost + ...
            50.0 * limit_violation_cost + ...
            cmd_activity * (0.90 * directional_gain_cost + 0.20 * isotropic_gain_cost) + ...
            0.02 * regularization_cost;

        if cost < best_cost
            best_cost = cost;
            lambda_star = lam;
        end
    end

    lambda_star = max(0, min(lambda_max, lambda_star));
end

% ----------------------------------------------------------------
function feat = make_nn_features( ...
    q, J, U, sv, w, v_cmd, e_p, q_min, q_max, v_cmd_max)

    sv = sv(:);

    sv_min = sv(end);
    sv_max = sv(1);

    cond_ratio = sv_min / (sv_max + 1e-10);

    v_norm = norm(v_cmd);

    if v_norm < 1e-12
        proj = zeros(3,1);
    else
        proj = abs(U' * v_cmd) / (v_norm + 1e-12);
        proj = min(proj, 1);
    end

    amp_index = norm((U' * v_cmd) ./ (sv + 1e-6)) / (v_norm + 1e-12);
    amp_index = log10(1 + amp_index);

    range_q = q_max - q_min;

    q_center = 0.5 * (q_min + q_max);
    half_range = 0.5 * range_q;

    q_norm = (q - q_center) ./ (half_range + 1e-12);
    q_norm = max(-1, min(1, q_norm));

    margin_each = min(q - q_min, q_max - q) ./ (half_range + 1e-12);
    margin_each = max(0, min(1, margin_each));

    margin_min = min(margin_each);
    near_limit_index = max(0, 0.20 - margin_min) / 0.20;

    e_norm = min(norm(e_p), 0.05) / 0.05;

    feat = [ ...
        log10(w + 1e-12), ...
        sv(1), ...
        sv(2), ...
        sv(3), ...
        cond_ratio, ...
        proj(1), ...
        proj(2), ...
        proj(3), ...
        amp_index, ...
        v_norm / v_cmd_max, ...
        e_norm, ...
        margin_min, ...
        near_limit_index, ...
        q_norm(:)', ...
        margin_each(:)' ...
    ];
end

% ----------------------------------------------------------------
function dq_null = joint_center_velocity(q, q_min, q_max, k_null)

    q_center   = 0.5 * (q_min + q_max);
    half_range = 0.5 * (q_max - q_min);

    dq_null = k_null * (q_center - q) ./ (half_range + 1e-12);
    dq_null = max(min(dq_null, 0.8), -0.8);
end

% ----------------------------------------------------------------
function Y_hat = nn_forward(X, W, b, n_layers)

    A = X;

    for l = 1:n_layers-1
        A = max(0, W{l} * A + b{l});
    end

    Z_out = W{n_layers} * A + b{n_layers};
    Z_out = max(min(Z_out, 60), -60);

    Y_hat = 1 ./ (1 + exp(-Z_out));
end

% ----------------------------------------------------------------
function lambda_yosh = yoshikawa_base_lambda(w, w_threshold, lambda_max)

    if w < w_threshold
        ratio = max(0, 1 - w / w_threshold);
        lambda_yosh = lambda_max * (1 - exp(-5 * ratio));
    else
        lambda_yosh = 0;
    end

    lambda_yosh = max(0, min(lambda_max, lambda_yosh));
end