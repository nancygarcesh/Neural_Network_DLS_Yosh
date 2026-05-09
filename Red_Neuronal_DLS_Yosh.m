clear;
clc;
close all;

%definicion de cada eslabon con parametros DH modificados del Dobot CR3
%mostrado en clase
L(1) = Link('modified','d',0,     'a',0,    'alpha',0,     'offset',0);
L(2) = Link('modified','d',0.128, 'a',0,    'alpha',pi/2,  'offset',pi/2);
L(3) = Link('modified','d',0.1165,'a',0.274,'alpha',pi,    'offset',0);
L(4) = Link('modified','d',0.116, 'a',0.230,'alpha',pi,    'offset',-pi/2);
L(5) = Link('modified','d',0.116, 'a',0,    'alpha',-pi/2, 'offset',0);
L(6) = Link('modified','d',0.105, 'a',0,    'alpha',pi/2,  'offset',0);


bot = SerialLink(L, 'name','Dobot-CR3');

deg = pi/180;



%BLOQUE A parametros

lambda_max  = 0.12;

dt          = 0.005;

%este es el limite de velocidad articular
dq_max      = 180 * deg;

alpha_gain  = 10.0;
%puntos por segmento de trayectoria
steps_seg   = 400;

%velocidad cartesiana maxima
v_cmd_max   = 0.25;

eye3 = eye(3);
eye6 = eye(6);

q_min = [-360, -360, -155, -360, -360, -360]' * deg;
q_max = [ 360,  360,  155,  360,  360,  360]' * deg;

k_null = 0.8;

%umbral de manipulabilidad para activar Yoshikawa en el modo YOSHREF
w_threshold_ref = 0.022;

use_stress_test = false;
use_yoshikawa_floor = false;

fprintf('CONTROL DLS CON NN Y YOSHIKAWA\n');



%BLOQUE B generacion de datos de entrenamiento

fprintf('Generando datos de entrenamiento\n');

N_samples = 60000;
rng(42);
%features de entrada para la nn
n_features = 25;

X_all = zeros(N_samples, n_features);
Y_all = zeros(N_samples, 1);

%grilla de 101 lambdas que el teacher evalua para encontrar el optimo
lambda_grid = linspace(0, lambda_max, 101);

idx = 0;
attempts = 0;
max_attempts = N_samples * 6; %limite de intentos para evitar bucle infinito


%generacion de las muestras aleatorias
while idx < N_samples && attempts < max_attempts
    attempts = attempts + 1;

    %postura articular aleatoria dentro de los limites
    q = q_min + (q_max - q_min) .* rand(6,1);

    J_full = bot.jacob0(q');
    J      = J_full(1:3,:);

    %SVD del jacobiano para obtener valores singulares
    [U,S,~] = svd(J,'econ');
    sv = diag(S);

    if numel(sv) < 3
        continue;
    end

    if any(~isfinite(sv)) || sv(1) < 1e-10
        continue;
    end

    %producto de los tres valores singulares para el calculo de la
    %manipulabilidad
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

    %llamada al teacher busca el lambda optimo por busqueda en grilla
    lambda_star = teacher_optimal_lambda_independent( ...
        J, U, sv, q, v_cmd, norm(e_p), ...
        q_min, q_max, lambda_grid, dq_max, dt, ...
        eye3, eye6, lambda_max, k_null, v_cmd_max);

    %construccion del vector de 25 features para la red
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
X_all = X_all(1:idx,:);
Y_all = Y_all(1:idx);

fprintf('Muestras generadas: %d / %d intentos\n', idx, attempts);
fprintf('lambda aprox 0:        %d  (%.1f%%)\n', ...
        sum(Y_all < 0.001), 100*sum(Y_all < 0.001)/idx);
fprintf('lambda adaptativo > 0: %d  (%.1f%%)\n', ...
        sum(Y_all >= 0.001), 100*sum(Y_all >= 0.001)/idx);
fprintf('lambda medio:          %.5f\n', mean(Y_all));
fprintf('lambda maximo:         %.5f\n', max(Y_all));


%mezcla aleatoria para evitar sesgos en el split
perm = randperm(idx);
X_all = X_all(perm,:);
Y_all = Y_all(perm);

%division 70% train 15% validacion 15% test
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
sig_X = std(X_raw_tr, 0, 1) + 1e-8; %evita division por cero

X_tr   = ((X_raw_tr   - mu_X) ./ sig_X)';
X_val  = ((X_raw_val  - mu_X) ./ sig_X)';
X_test = ((X_raw_test - mu_X) ./ sig_X)';

Y_tr   = (Y_raw_tr   / lambda_max)';
Y_val  = (Y_raw_val  / lambda_max)';
Y_test = (Y_raw_test / lambda_max)';

fprintf('\nSplit dataset:\n');
fprintf('Train: %d | Val: %d | Test: %d\n', n_train, n_val, n_test);



%BLOQUE C arquitectura y entrenamiento de la NN

fprintf('\nEntrenando red neuronal\n');

%arquitectura de 25 entradas -> 128 -> 256 -> 128 -> 64 -> 1 salida
arch = [n_features, 128, 256, 128, 64, 1];
n_layers = length(arch) - 1;

W = cell(n_layers,1);
b = cell(n_layers,1);

for l = 1:n_layers
    fan_in = arch(l);
    fan_out = arch(l+1);
    %inicializacion para activaciones ReLU
    W{l} = randn(fan_out, fan_in) * sqrt(2 / fan_in);
    b{l} = zeros(fan_out, 1);
end

lr         = 1e-3; %learning rate
lr_decay   = 0.92; %factor de decaimiento cada 15 epochs
n_epochs   = 200; %maximo de epochs
batch_size = 512; %tamano de mini-batch

beta1      = 0.9;
beta2      = 0.999;
eps_adam   = 1e-8;

mW = cell(n_layers,1); vW = cell(n_layers,1);
mb = cell(n_layers,1); vb = cell(n_layers,1);

for l = 1:n_layers
    mW{l} = zeros(size(W{l})); vW{l} = zeros(size(W{l}));
    mb{l} = zeros(size(b{l})); vb{l} = zeros(size(b{l}));
end

%historial de perdida para graficar curvas de entrenamiento y observar la
%comparacion entre train y validation
loss_hist = zeros(n_epochs, 1);
val_hist  = zeros(n_epochs, 1);

%control de early stopping
t_adam     = 0;
best_val   = inf;
best_W     = W;
best_b     = b;
patience   = 25;
no_improve = 0;
ep_done    = 0;

fprintf('Epochs: %d | Batch: %d | LR: %.4f | Arq: ', n_epochs, batch_size, lr);
fprintf('%d-', arch); fprintf('\b \n\n');

%bucle de entrenamiento epoch por epoch
for ep = 1:n_epochs
    ep_done = ep;

    %mezcla aleatoria del conjunto de entrenamiento en cada epoch
    perm_ep = randperm(n_train);
    X_sh = X_tr(:, perm_ep);
    Y_sh = Y_tr(perm_ep);

    n_batches = ceil(n_train / batch_size);
    ep_loss = 0;
    used_batches = 0;

    %recorre cada mini-batch
    for b_idx = 1:n_batches
        i1 = (b_idx-1)*batch_size + 1;
        i2 = min(b_idx*batch_size, n_train);
        bs = i2 - i1 + 1;

        Xb = X_sh(:, i1:i2);
        Yb = Y_sh(i1:i2);

        A_cache = cell(n_layers+1, 1);
        Z_cache = cell(n_layers, 1);
        A_cache{1} = Xb;

        %capas ocultas con activacion ReLU
        for l = 1:n_layers-1
            Z_cache{l}   = W{l} * A_cache{l} + b{l};
            A_cache{l+1} = max(0, Z_cache{l});
        end

        %capa de salida con sigmoide para obtener valor en [0,1] o entre 0
        %a 0,12
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

        %actualizacion de pesos con Adam para cada capa
        for l = 1:n_layers
            mW{l} = beta1*mW{l} + (1-beta1)*dW_acc{l};
            vW{l} = beta2*vW{l} + (1-beta2)*dW_acc{l}.^2;

            mb{l} = beta1*mb{l} + (1-beta1)*dB_acc{l};
            vb{l} = beta2*vb{l} + (1-beta2)*dB_acc{l}.^2;

            W{l} = W{l} - lr * (mW{l}/bc1) ./ (sqrt(vW{l}/bc2) + eps_adam);
            b{l} = b{l} - lr * (mb{l}/bc1) ./ (sqrt(vb{l}/bc2) + eps_adam);
        end
    end

    %evaluacion sobre validacion al final del epoch
    Y_val_hat = nn_forward(X_val, W, b, n_layers);
    val_loss  = mean((Y_val_hat - Y_val).^2);

    loss_hist(ep) = ep_loss / used_batches;
    val_hist(ep)  = val_loss;

    %guarda el mejor modelo por validacion
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
        fprintf('  Epoch %3d/%d | Loss: %.6f | Val: %.6f | MAE lam: %.5f | LR: %.6f\n', ...
                ep, n_epochs, loss_hist(ep), val_hist(ep), val_mae_lam, lr);
    end

    %early stopping si no mejora durante patience epochs
    if no_improve >= patience
        fprintf('Early stopping en epoch %d\n', ep);
        break;
    end
end

W = best_W;
b = best_b;

Y_test_hat = nn_forward(X_test, W, b, n_layers);
test_mse = mean((Y_test_hat - Y_test).^2);
test_mae_lambda = mean(abs(Y_test_hat - Y_test)) * lambda_max;

fprintf('\nEntrenamiento completado\n');
fprintf('Mejor Val MSE  : %.8f\n', best_val);
fprintf('Test MSE       : %.8f\n', test_mse);
fprintf('Test MAE lambda: %.6f\n', test_mae_lambda);


%BLOQUE D curva de entrenamiento (grafico)

figure('Name','Entrenamiento NN','Color','k','Position',[20 60 680 360]);
ax_loss = axes('Color','k','XColor','w','YColor','w');
hold(ax_loss,'on'); grid(ax_loss,'on');

valid_ep = 1:ep_done;
%curva de perdida de entrenamiento
plot(ax_loss, valid_ep, loss_hist(valid_ep), '-', 'Color',[0.2 0.8 1], 'LineWidth',1.8);
%curva de perdida de validacion
plot(ax_loss, valid_ep, val_hist(valid_ep),  '-', 'Color',[1 0.5 0.1], 'LineWidth',1.8);

xlabel(ax_loss,'Epoch','Color','w');
ylabel(ax_loss,'MSE','Color','w');
title(ax_loss,'Curva de entrenamiento','Color','c','FontSize',12,'FontWeight','bold');
legend(ax_loss,'Train','Validacion','TextColor','w','Color','k');


% BLOQUE E construccion de trayectoria

fprintf('\nConstruyendo trayectoria\n');

%altura fija en z donde se dibuja la figura asignada en la clase
altura = 0.20;
P_all = build_letter_p_trajectory(steps_seg, use_stress_test);
N_total = size(P_all,1);
P3_all = [P_all, altura*ones(N_total,1)];
v_ff_all = zeros(3, N_total);
v_ff_all(:,2:end) = (P3_all(2:end,:) - P3_all(1:end-1,:))' / dt;
v_norm_all = sqrt(sum(v_ff_all.^2, 1));
scale_v = min(1, v_cmd_max ./ (v_norm_all + 1e-12));
v_ff_all = v_ff_all .* scale_v;


%mascara para solo controlar posicion xyz no la orientacion
pt = @(x,y,z) transl(x,y,z) * trotx(0);
maskXYZ = [1 1 1 0 0 0];

T0 = pt(P_all(1,1), P_all(1,2), altura);

%cinematica inversa para obtener q0 desde la pose inicial
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

fprintf('q0 usado:\n');
disp(q0');


% BLOQUE F estructura de parametros y red


%estructura con todos los parametros del controlador para pasarlos a las simulaciones
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
params.lambda_const    = mean(Y_raw_tr); %lambda promedio para el modo CONST

params.live_target_frames = 60; %frames por segundo objetivo para la visualizacion en vivo
params.live_fast_draw = true;

%estructura de la red neuronal con pesos normalizacion y suavizado
net.W          = W;
net.b          = b;
net.n_layers   = n_layers;
net.mu_X       = mu_X;
net.sig_X      = sig_X;
net.beta_lam   = 0.60;
net.lambda_max = lambda_max;

%BLOQUE G simulaciones comparativas


fprintf('\nEjecutando simulaciones comparativas\n');

%ZERO: lambda=0, pseudoinversa sin amortiguamiento referencia base
res_zero  = simulate_controller(bot, P_all, altura, v_ff_all, q0, params, net, 'ZERO',    false);
%CONST: lambda fijo igual al promedio aprendido
res_const = simulate_controller(bot, P_all, altura, v_ff_all, q0, params, net, 'CONST',   false);
%YOSHREF: Yoshikawa clasico basado en manipulabilidad
res_yosh  = simulate_controller(bot, P_all, altura, v_ff_all, q0, params, net, 'YOSHREF', false);

%NN: propuesta principal lambda predicho por la red neuronal
res_nn    = simulate_controller(bot, P_all, altura, v_ff_all, q0, params, net, 'NN',      true);



% BLOQUE H graficos de resultados finales

scr = get(0,'ScreenSize');

%figura de lambda comparativo entre controladores
figure('Name','Lambda comparativo','Color','k','Position',[800,scr(4)-420,680,340]);
ax2 = axes('Color','k','XColor','w','YColor','w');
hold(ax2,'on'); grid(ax2,'on');

plot(ax2, 1:N_total, res_nn.lambda_hist, '-', 'Color',[0.8 0.2 1], 'LineWidth',2.0);
plot(ax2, 1:N_total, res_yosh.lambda_hist, '--', 'Color',[0.2 0.9 1], 'LineWidth',1.4);
plot(ax2, 1:N_total, res_const.lambda_hist, '-.', 'Color',[0.9 0.9 0.9], 'LineWidth',1.2);

xlabel(ax2,'Paso','Color','w');
ylabel(ax2,'lambda','Color','w');
title(ax2,'Lambda aplicado: NN vs referencias','Color','m','FontSize',12,'FontWeight','bold');
legend(ax2,{'NN sin piso','Yosh ref','Constante'}, ...
       'TextColor','w','Color','k','Location','northeast');
ylim(ax2,[0, lambda_max*1.15]);

%figura de error cartesiano comparativo
figure('Name','Error cartesiano','Color','k','Position',[800,scr(4)-790,680,340]);
ax3 = axes('Color','k','XColor','w','YColor','w');
hold(ax3,'on'); grid(ax3,'on');

plot(ax3, 1:N_total, res_zero.err_hist, '-', 'Color',[1 0.25 0.25], 'LineWidth',1.2);
plot(ax3, 1:N_total, res_const.err_hist, '-', 'Color',[0.9 0.9 0.9], 'LineWidth',1.2);
plot(ax3, 1:N_total, res_yosh.err_hist, '-', 'Color',[0.2 0.9 1], 'LineWidth',1.2);
plot(ax3, 1:N_total, res_nn.err_hist, '-', 'Color',[0.8 0.2 1], 'LineWidth',1.8);

xlabel(ax3,'Paso','Color','w');
ylabel(ax3,'Error [m]','Color','w');
title(ax3,'Error cartesiano comparativo','Color','y','FontSize',12,'FontWeight','bold');
legend(ax3,{'lambda=0','Constante','Yosh ref','NN sin piso'}, ...
       'TextColor','w','Color','k','Location','northeast');


%figura de ganancia inversa efectiva del DLS
figure('Name','Ganancia inversa DLS','Color','k','Position',[800,80,680,340]);
ax5 = axes('Color','k','XColor','w','YColor','w');
hold(ax5,'on'); grid(ax5,'on');

plot(ax5, 1:N_total, res_zero.inv_gain_max_hist, '-', 'Color',[1 0.25 0.25], 'LineWidth',1.2);
plot(ax5, 1:N_total, res_yosh.inv_gain_max_hist, '-', 'Color',[0.2 0.9 1], 'LineWidth',1.2);
plot(ax5, 1:N_total, res_nn.inv_gain_max_hist, '-', 'Color',[0.8 0.2 1], 'LineWidth',1.8);

xlabel(ax5,'Paso','Color','w');
ylabel(ax5,'max sigma/(sigma^2+lambda^2)','Color','w');
title(ax5,'Ganancia inversa efectiva','Color','c','FontSize',12,'FontWeight','bold');
legend(ax5,{'lambda=0','Yosh ref','NN sin piso'}, ...
       'TextColor','w','Color','k','Location','northeast');


% BLOQUE I resumen cuantitativo comparativo

fprintf('\nRESUMEN COMPARATIVO:\n');
print_metrics('ZERO lambda=0', res_zero, N_total);
print_metrics('CONST lambda media', res_const, N_total);
print_metrics('YOSH referencia', res_yosh, N_total);
print_metrics('NN sin piso', res_nn, N_total);

fprintf('\nAPORTE REAL DE LA NN:\n');

%reduccion porcentual de dq_max frente a lambda=0
dq_reduction = 100 * (res_zero.metrics.dq_raw_max - res_nn.metrics.dq_raw_max) / ...
               (abs(res_zero.metrics.dq_raw_max) + 1e-12);

%reduccion porcentual de ganancia inversa frente a lambda=0
gain_reduction = 100 * (res_zero.metrics.inv_gain_max - res_nn.metrics.inv_gain_max) / ...
                 (abs(res_zero.metrics.inv_gain_max) + 1e-12);

%cambio porcentual de error medio frente a lambda=0
err_change = 100 * (res_nn.metrics.err_mean - res_zero.metrics.err_mean) / ...
             (abs(res_zero.metrics.err_mean) + 1e-12);

%mejora porcentual del score global frente a lambda=0
score_change = 100 * (res_zero.metrics.score - res_nn.metrics.score) / ...
               (abs(res_zero.metrics.score) + 1e-12);

fprintf('Reduccion dq_max vs lambda=0              : %.2f %%\n', dq_reduction);
fprintf('Reduccion ganancia inversa max vs lambda=0: %.2f %%\n', gain_reduction);
fprintf('Cambio error medio vs lambda=0            : %.2f %%\n', err_change);
fprintf('Mejora score global vs lambda=0           : %.2f %%\n', score_change);

%si la NN no supera a lambda=0 se mostrara explicitamente
if res_nn.metrics.score < res_zero.metrics.score
    fprintf('La NN aporta mejora segun el score global definido.\n');
else
    fprintf('En esta trayectoria, la NN NO supera a lambda=0 segun el score global.\n');
    fprintf('Esto no se oculta: el resultado queda reportado honestamente.\n');
end

%correlacion entre lambda de la NN y lambda de Yoshikawa
corr_mat = corrcoef(res_nn.lambda_hist, res_yosh.lambda_hist);
if all(isfinite(corr_mat(:))) && size(corr_mat,1)==2
    fprintf('Correlacion lambda_NN vs lambda_Yosh_ref   : %.4f\n', corr_mat(1,2));
else
    fprintf('Correlacion lambda_NN vs lambda_Yosh_ref   : no definida\n');
end


%FUNCIONES AUXILIARES

%genera la trayectoria 2D de la figura por segmentos y arcos
function P_all = build_letter_p_trajectory(steps_seg, use_stress_test)

    %vertices de la figura definidos en coordenadas xy en metros
    A=[0.15,0.15];  B=[0.15,0.40];  C=[0.175,0.425];
    D=[0.475,0.425]; F=[0.505,0.275]; G=[0.505,0.15];
    H=[0.41,0.15];  I=[0.335,0.15];
    Jp=[0.26,0.15]; K=[0.26,0.24];

    %funcion anonima para crear segmentos lineales entre dos puntos
    mk = @(p1,p2,n) [linspace(p1(1),p2(1),n)', ...
                     linspace(p1(2),p2(2),n)'];

    segs = {};

    %segmentos rectos de la parte vertical de la P
    segs{end+1} = mk(A,B,steps_seg);
    segs{end+1} = mk(B,C,steps_seg);
    segs{end+1} = mk(C,D,steps_seg);

    %parametros del arco superior derecho
    centroDer = [0.5049936661454, 0.3529987332291];
    radioDer  = sqrt(0.0060838024255);

    %angulos de inicio y fin del arco superior
    angD = atan2(D(2)-centroDer(2), D(1)-centroDer(1));
    angF = atan2(F(2)-centroDer(2), F(1)-centroDer(1));

    if angF < angD
        angF = angF + 2*pi;
    end

    angArcDer = linspace(angD, angF, steps_seg);

    %puntos del arco superior
    arc_der = [centroDer(1)+radioDer*cos(angArcDer(:)), ...
               centroDer(2)+radioDer*sin(angArcDer(:))];

    segs{end+1} = arc_der;
    segs{end+1} = mk(F,G,steps_seg);
    segs{end+1} = mk(G,H,steps_seg);

    %arco inferior de la panza como semicirculo
    centro_inf = 0.5 * (H + Jp);
    radio_inf  = norm(H - Jp) / 2;

    ang_H  = atan2(H(2)  - centro_inf(2), H(1)  - centro_inf(1));
    ang_Jp = atan2(Jp(2) - centro_inf(2), Jp(1) - centro_inf(1));

    if ang_Jp > ang_H
        ang_Jp = ang_Jp - 2*pi;
    end

    angArcInf = linspace(ang_H, ang_Jp, steps_seg);

    %puntos del arco inferior
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

    %agrega cada segmento sin el primer punto para evitar duplicados
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


%simular un paso de control DLS con el modo de lambda indicado
function res = simulate_controller(bot, P_all, altura, v_ff_all, q0, params, net, mode, live_plot)

    if nargin < 9
        live_plot = false;
    end

    N_total = size(P_all,1);

    q = q0(:); %postura articular inicial

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
    draw_timer = [];
    draw_time = 0;
    active_x = NaN(1,N_total);
    active_y = NaN(1,N_total);
    active_z = NaN(1,N_total);
    active_count = 0;

    if isfield(params,'live_target_frames')
        update_vis = max(1, ceil(N_total / params.live_target_frames));
    else
        update_vis = max(1, ceil(N_total / 60));
    end

    if live_plot
        draw_timer = tic;

        scr = get(0,'ScreenSize');

        live.fig = figure( ...
            'Name','Dibujo en tiempo real - NN DLS', ...
            'Color','k', ...
            'Position',[30, 80, min(1450,scr(3)-80), min(760,scr(4)-140)]);

        try
            set(live.fig,'Renderer','opengl');
        catch
        end

        try
            set(live.fig,'GraphicsSmoothing','off');
        catch
        end

        live.tl = tiledlayout(live.fig,2,3, ...
            'TileSpacing','compact', ...
            'Padding','compact');

        live.ax_traj = nexttile(live.tl,[2 2]);
        hold(live.ax_traj,'on');
        grid(live.ax_traj,'on');
        axis(live.ax_traj,'equal');

        set(live.ax_traj, ...
            'Color','k', ...
            'XColor','w', ...
            'YColor','w', ...
            'ZColor','w', ...
            'GridColor',[0.35 0.35 0.35]);

        xlabel(live.ax_traj,'X [m]','Color','w');
        ylabel(live.ax_traj,'Y [m]','Color','w');
        zlabel(live.ax_traj,'Z [m]','Color','w');

        title(live.ax_traj, ...
            'Trayectoria en tiempo real - Control DLS con NN', ...
            'Color','c', ...
            'FontSize',14, ...
            'FontWeight','bold');

        view(live.ax_traj,55,28);

        %trayectoria deseada de fondo
        live.h_des = plot3(live.ax_traj, ...
            P_all(:,1), ...
            P_all(:,2), ...
            altura*ones(N_total,1), ...
            '--', ...
            'Color',[0.75 0.75 0.75], ...
            'LineWidth',3.0);

        %trayectoria ejecutada por la NN que se actualiza en tiempo real
        live.h_exec = plot3(live.ax_traj, ...
            NaN, NaN, NaN, ...
            '-', ...
            'Color',[0.75 0.10 1.00], ...
            'LineWidth',4.0);

        %puntos donde lambda > 0.001, indicando DLS activo
        live.h_active = plot3(live.ax_traj, ...
            NaN, NaN, NaN, ...
            'o', ...
            'Color',[1.00 0.15 0.15], ...
            'MarkerFaceColor',[1.00 0.15 0.15], ...
            'MarkerSize',5);

        %marcador del punto actual del robot
        live.h_now = plot3(live.ax_traj, ...
            NaN, NaN, NaN, ...
            'p', ...
            'Color',[0.10 1.00 0.30], ...
            'MarkerFaceColor',[0.10 1.00 0.30], ...
            'MarkerSize',14, ...
            'LineWidth',2.2);

        legend(live.ax_traj, ...
            [live.h_des live.h_exec live.h_active live.h_now], ...
            {'Deseada','Ejecutada NN','DLS activo','Punto actual'}, ...
            'TextColor','w', ...
            'Color','k', ...
            'Location','northeast');

        xlim(live.ax_traj,[min(P_all(:,1))-0.05, max(P_all(:,1))+0.08]);
        ylim(live.ax_traj,[min(P_all(:,2))-0.08, max(P_all(:,2))+0.08]);
        zlim(live.ax_traj,[altura-0.08, altura+0.08]);

        %cuadro de estado en tiempo real con metricas del paso actual
        live.status = annotation(live.fig,'textbox', ...
            [0.025 0.83 0.28 0.13], ...
            'String','Inicializando...', ...
            'Color','w', ...
            'BackgroundColor',[0.05 0.05 0.05], ...
            'EdgeColor',[0.2 0.8 1.0], ...
            'LineWidth',1.5, ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'FitBoxToText','off');

        %axes de lambda en tiempo real
        live.ax_lam = nexttile(live.tl,3);
        hold(live.ax_lam,'on');
        grid(live.ax_lam,'on');

        set(live.ax_lam, ...
            'Color','k', ...
            'XColor','w', ...
            'YColor','w', ...
            'GridColor',[0.35 0.35 0.35]);

        title(live.ax_lam,'Lambda en tiempo real','Color','m','FontSize',12,'FontWeight','bold');
        xlabel(live.ax_lam,'Paso','Color','w');
        ylabel(live.ax_lam,'\lambda','Color','w');

        %lambda suavizado que realmente se aplica al DLS
        live.h_lam_nn = plot(live.ax_lam,NaN,NaN, ...
            '-', ...
            'Color',[0.85 0.15 1.00], ...
            'LineWidth',3.2);

        live.h_lam_raw = plot(live.ax_lam,NaN,NaN, ...
            '--', ...
            'Color',[1.00 0.65 0.15], ...
            'LineWidth',1.8);

        %lambda de Yoshikawa como referencia visual
        live.h_lam_yosh = plot(live.ax_lam,NaN,NaN, ...
            ':', ...
            'Color',[0.10 0.90 1.00], ...
            'LineWidth',2.3);

        legend(live.ax_lam, ...
            [live.h_lam_nn live.h_lam_raw live.h_lam_yosh], ...
            {'NN aplicado','NN crudo','Yosh ref'}, ...
            'TextColor','w', ...
            'Color','k', ...
            'Location','northeast');

        ylim(live.ax_lam,[0 params.lambda_max*1.15]);
        xlim(live.ax_lam,[1 N_total]);
        

        fast_drawnow();
    end


    %bucle principal de control un paso por punto de trayectoria
    for k = 1:N_total

        %punto deseado
        p_d = [P_all(k,1); P_all(k,2); altura];

        %cinematica directa para obtener posicion actual del efector
        T_cur = bot.fkine(q');
        p_cur = T_cur.t;

        %error cartesiano de posicion
        e_p = p_d - p_cur;

        %jacobiano (3x6)
        J_full = bot.jacob0(q');
        J      = J_full(1:3,:);

        %SVD del jacobiano: U, valores singulares, manipulabilidad
        [U,S,~] = svd(J,'econ');
        sv = diag(S);
        w  = prod(sv); %manipulabilidad: producto de valores singulares

        %ganancia adaptativa aumenta cuando el error es mayor
        gain_k = params.alpha_gain * (1 + 2 * min(norm(e_p), 0.05) / 0.05);
        v_cmd = v_ff_all(:,k) + gain_k * e_p;

        %saturacion de velocidad comandada
        nv = norm(v_cmd);
        if nv > params.v_cmd_max
            v_cmd = v_cmd * (params.v_cmd_max / nv);
        end

        lambda_yosh = yoshikawa_base_lambda(w, params.w_threshold_ref, params.lambda_max);

        switch upper(mode)
            case 'NN'
                feat = make_nn_features(q, J, U, sv, w, v_cmd, e_p, ...
                                        params.q_min, params.q_max, params.v_cmd_max);

                %normaliza con estadisticas del conjunto de entrenamiento
                feat_norm = ((feat - net.mu_X) ./ net.sig_X)';

                lambda_pred = nn_forward(feat_norm, net.W, net.b, net.n_layers);
                lambda_pred = max(0, min(1, lambda_pred));
                lam_raw = lambda_pred * net.lambda_max;
                lam_raw = max(0, min(params.lambda_max, lam_raw));

                lam = net.beta_lam * lam_prev + (1 - net.beta_lam) * lam_raw;
                lam = max(0, min(params.lambda_max, lam));
                lam_prev = lam;

            case 'ZERO'
                %pseudoinversa sin amortiguamiento: caso base
                lam_raw = 0;
                lam = 0;

            case 'CONST'
                %lambda fijo igual al promedio aprendido en entrenamiento
                lam_raw = params.lambda_const;
                lam = params.lambda_const;

            case 'YOSHREF'
                %lambda segun manipulabilidad de Yoshikawa
                lam_raw = lambda_yosh;
                lam = lambda_yosh;

            otherwise
                error('Modo de controlador no reconocido');
        end

        %calculo del Jacobiano pseudoinverso amortiguado DLS
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
        v_abs_err_hist(k)  = norm(v_cmd - v_real); %error de velocidad cartesiana

        if v_cmd_norm_hist(k) > v_cmd_eps
            v_rel_err_hist(k) = v_abs_err_hist(k) / v_cmd_norm_hist(k);
        end

        q = q + dq * params.dt;

        q_low_viol  = max(0, params.q_min - q);
        q_high_viol = max(0, q - params.q_max);
        q_limit_viol(k) = max([q_low_viol; q_high_viol; 0]);

        %cinematica directa para obtener posicion real tras el paso de control
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

        if live_plot

            %registra el punto si lambda estuvo activo
            if lambda_hist(k) > 0.001
                active_count = active_count + 1;
                active_x(active_count) = pos_hist(1,k);
                active_y(active_count) = pos_hist(2,k);
                active_z(active_count) = pos_hist(3,k);
            end

            if mod(k,update_vis)==0 || k==1 || k==N_total

                %actualiza trayectoria ejecutada
                set(live.h_exec, ...
                    'XData',pos_hist(1,1:k), ...
                    'YData',pos_hist(2,1:k), ...
                    'ZData',pos_hist(3,1:k));

                %actualiza marcador del punto actual
                set(live.h_now, ...
                    'XData',pos_hist(1,k), ...
                    'YData',pos_hist(2,k), ...
                    'ZData',pos_hist(3,k));

                %actualiza puntos de lambda activo
                if active_count > 0
                    set(live.h_active, ...
                        'XData',active_x(1:active_count), ...
                        'YData',active_y(1:active_count), ...
                        'ZData',active_z(1:active_count));
                end

                %actualiza grafica de lambda
                set(live.h_lam_nn, ...
                    'XData',1:k, ...
                    'YData',lambda_hist(1:k));

                set(live.h_lam_raw, ...
                    'XData',1:k, ...
                    'YData',lambda_raw_hist(1:k));

                set(live.h_lam_yosh, ...
                    'XData',1:k, ...
                    'YData',lambda_yosh_hist(1:k));

                

                elapsed_draw = toc(draw_timer);
                progress_pct = 100 * k / N_total;

                status_txt = sprintf([ ...
                    'Tiempo dibujo: %.2f s\n' ...
                    'Paso: %d / %d   (%.1f %%)\n' ...
                    'Refresco grafico cada: %d pasos\n' ...
                    'lambda NN: %.5f\n' ...
                    'Error pos: %.6f m\n' ...
                    'max |dq|: %.4f rad/s\n' ...
                    'Ganancia inv: %.4f'], ...
                    elapsed_draw, ...
                    k, N_total, progress_pct, ...
                    update_vis, ...
                    lambda_hist(k), ...
                    err_hist(k), ...
                    dq_abs_max_hist(k), ...
                    inv_gain_max_hist(k));

                set(live.status,'String',status_txt);

                fast_drawnow();
            end
        end
    end

    sim_time = toc(sim_timer);

    if live_plot
        draw_time = toc(draw_timer);
    end

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

    %calculo de metricas resumidas del controlador
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
    res.metrics.draw_time      = draw_time;
    res.metrics.avg_step_time  = sim_time / N_total;

    %score global: ponderacion de error vel, esfuerzo, saturacion, limites y lambda
    res.metrics.score = ...
        rel_score + ...
        0.03 * mean((dq_abs_max_hist / params.dq_max).^2) + ...
        5.00 * mean(dq_sat_hist > 0) + ...
        50.0 * max(q_limit_viol)^2 + ...
        0.02 * mean(lambda_hist / params.lambda_max);
end


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

    if res.metrics.draw_time > 0
        fprintf('Tiempo dibujo trayectoria en vivo        : %.3f s\n', res.metrics.draw_time);
    end
end


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

        %calcula DLS con este lambda
        A = JJT + (lam^2 + 1e-10) * eye3;
        J_dls = J' * (A \ eye3);

        dq_task = J_dls * v_cmd;
        N_null  = eye6 - J_dls * J;
        dq      = dq_task + N_null * dq_null;

        %aplica saturacion para simular el comportamiento real
        dq_sat = max(min(dq, dq_max), -dq_max);
        v_real = J * dq_sat;

        %C_track: penaliza perdida de seguimiento cartesiano
        tracking_cost = norm(v_cmd - v_real)^2 / v_norm2;

        %C_joint: penaliza velocidades articulares grandes
        joint_speed_cost = mean((dq / dq_max).^2);

        %C_sat: penaliza fuertemente la saturacion articular
        saturation_cost = mean((max(0, abs(dq) - dq_max) / dq_max).^2);

        q_next = q + dq_sat * dt;

        margin_next = min(q_next - q_min, q_max - q_next) ./ (0.5 * range_q);
        margin_next = max(-1, min(1, margin_next));

        %C_limit: penaliza cercanía a los limites articulares
        near_limit_cost = mean(max(0, 0.12 - margin_next).^2) / (0.12^2);

        %penaliza violaciones reales de limites articulares
        violation_low  = max(0, q_min - q_next) ./ range_q;
        violation_high = max(0, q_next - q_max) ./ range_q;
        limit_violation_cost = sum(violation_low.^2 + violation_high.^2);

        inv_gain = sv ./ (sv.^2 + lam^2 + 1e-12);

        directional_gain = norm(inv_gain .* dir_weight);
        max_inv_gain     = max(inv_gain);

        %C_gain: penaliza cuando la ganancia excede el umbral seguro
        directional_gain_cost = max(0, (directional_gain - gain_safe) / (gain_safe + 1e-12))^2;
        isotropic_gain_cost   = max(0, (max_inv_gain     - gain_safe) / (gain_safe + 1e-12))^2;

        %C_reg: penaliza usar lambda innecesariamente alto
        regularization_cost = (lam / lambda_max)^2;

        cost = ...
            track_weight * tracking_cost + ...
            0.04 * joint_speed_cost + ...
            8.00 * saturation_cost + ...
            0.25 * near_limit_cost + ...
            50.0 * limit_violation_cost + ...
            cmd_activity * (0.90 * directional_gain_cost + 0.20 * isotropic_gain_cost) + ...
            0.02 * regularization_cost;

        %aqui guarda el lambda de menor costo
        if cost < best_cost
            best_cost = cost;
            lambda_star = lam;
        end
    end

    lambda_star = max(0, min(lambda_max, lambda_star));
end


%funcion que construye el vector de 25 features de entrada a la red

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
        log10(w + 1e-12), ...   %1: log de manipulabilidad
        sv(1), ...              %2: valor singular mayor
        sv(2), ...              %3: valor singular medio
        sv(3), ...              %4: valor singular menor
        cond_ratio, ...         %5: razon de condicion
        proj(1), ...            %6: proyeccion del cmd en dir singular 1
        proj(2), ...            %7: proyeccion del cmd en dir singular 2
        proj(3), ...            %8: proyeccion del cmd en dir singular 3
        amp_index, ...          %9: indice de amplificacion logaritmico
        v_norm / v_cmd_max, ... %10: magnitud del comando normalizada
        e_norm, ...             %11: error cartesiano normalizado
        margin_min, ...         %12: margen articular minimo
        near_limit_index, ...   %13: indicador de cercania a limite
        q_norm(:)', ...         %14-19: postura articular normalizada (6 valores)
        margin_each(:)' ...     %20-25: margen por articulacion (6 valores)
    ];
end


function dq_null = joint_center_velocity(q, q_min, q_max, k_null)

    q_center   = 0.5 * (q_min + q_max);
    half_range = 0.5 * (q_max - q_min);

    dq_null = k_null * (q_center - q) ./ (half_range + 1e-12);

    dq_null = max(min(dq_null, 0.8), -0.8);
end


%forward de la red neuronal

function Y_hat = nn_forward(X, W, b, n_layers)

    A = X;

    %capas ocultas con activacion ReLU
    for l = 1:n_layers-1
        A = max(0, W{l} * A + b{l});
    end

    %capa de salida con sigmoide para obtener salida en [0,1]
    Z_out = W{n_layers} * A + b{n_layers};
    Z_out = max(min(Z_out, 60), -60);

    Y_hat = 1 ./ (1 + exp(-Z_out)); %sigmoide
end


function lambda_yosh = yoshikawa_base_lambda(w, w_threshold, lambda_max)

    if w < w_threshold
        ratio = max(0, 1 - w / w_threshold);
        lambda_yosh = lambda_max * (1 - exp(-5 * ratio));
    else
        lambda_yosh = 0;
    end
    lambda_yosh = max(0, min(lambda_max, lambda_yosh));
end


function fast_drawnow()

    try
        drawnow limitrate nocallbacks;
    catch
        drawnow limitrate;
    end
end