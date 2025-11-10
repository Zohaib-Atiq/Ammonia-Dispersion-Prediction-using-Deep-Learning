# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

import tensorflow as tf
from tensorflow import keras
import tabulate
from tensorflow.keras import layers  # type: ignore
import keras_tuner as kt  # type: ignore

# %%
# ---- Load Dataset ----
data = pd.read_excel("unique_points.xlsx", sheet_name="Sheet1")

# Features
X = data[["Air Temperature (°C)", "Wind Speed (m/s)", "Hole Diameter (mm)"]]

# Targets (now includes threat zones)
y = data[['Indoor Concentration (ppm)', 'Outdoor Concentraton (ppm)',
          'Red Zone (miles) ', 'Orange Zone (miles) ', 'Yellow Zone (miles)']]

# %%
# ---- Scale Features and Targets ----
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# %%
# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# %% 
# ---- Hyperparameter Tuning using Keras Tuner ----
run_hyperparameter_tuning = False
if run_hyperparameter_tuning:
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))

        for i in range(hp.Int("num_layers", 2, 3)):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                activation='relu'
            ))
            model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', 0.0, 0.4, step=0.1)))

        model.add(layers.Dense(5))  # Output: 5 targets (Indoor & Outdoor + Threat Zones)

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
            ),
            loss="mse",
            metrics=["mae", "mse", keras.metrics.RootMeanSquaredError()]
        )

        return model

    # Initialize the tuner
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="tuner_dir",
        project_name="indoor_outdoor_conc"
    )

    # Run the tuner
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stop], verbose=1)

    # Retrieve best model and hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]

    model = tuner.hypermodel.build(best_hp)

    # Train best model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

# %%
# ---- Run the Best Model ----
print("Best Hyperparameters:")
# best_hp_values = print(best_hp.values)
best_hp_values = {
                'num_layers': 3,
                'units_0': 256,
                'dropout_0': 0.30000000000000004,
                'units_1': 160,
                'dropout_1': 0.1,
                'learning_rate': 0.00047590319258532097,
                'units_2': 160,
                'dropout_2': 0.0
                }
print(best_hp_values)
# ---- Build Keras Model Based on Best Hyperparameters ----
model = keras.Sequential()
for i in range(best_hp_values['num_layers']):
    model.add(layers.Dense(
        units=best_hp_values[f'units_{i}'],
        activation='relu'
    ))
    model.add(layers.Dropout(rate=best_hp_values[f'dropout_{i}']))

model.add(layers.Dense(5))  # Output: 5 targets (Indoor & Outdoor + Threat Zones)

# Compile the model with the best learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_hp_values['learning_rate']),
    loss="mse",
    metrics=["mae", "mse", keras.metrics.RootMeanSquaredError()]
)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train best model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# %%
# ---- Evaluate the Model ----
test_loss, test_mae, test_mse, test_rmse = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

# %%
# ---- Predict and Inverse Scale ----
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# %%
# ---- Performance Metrics for All Outputs ----
# output_names = ['Indoor (ppm)', 'Outdoor (ppm)']
output_names = ['Indoor (ppm)', 'Outdoor (ppm)', 'Red Zone (miles)', 'Orange Zone (miles)', 'Yellow Zone (miles)']



metrics_table = []
for i, name in enumerate(output_names):
    rmse = root_mean_squared_error(y_actual[:, i], y_pred[:, i])
    r2 = r2_score(y_actual[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_actual[:, i], y_pred[:, i])
    mape = (mae / np.mean(y_actual[:, i])) * 100

    metrics_table.append([name, f"{rmse:.4f}", f"{r2:.4f}", f"{mae:.4f}", f"{mape:.2f}%"])

table = pd.DataFrame(metrics_table, columns=["Output", "RMSE", "R²", "MAE", "MAPE"])
print(tabulate.tabulate(metrics_table, headers=["Output", "RMSE", "R²", "MAE", "MAPE"], tablefmt="github"))

# %%
# ---- Plot Loss Curve ----
plt.figure(figsize=(10, 5))

fontfamily = {'family': 'Times New Roman', 'weight': 'regular', 'size': 14}
plt.rc('font', **fontfamily)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

plt.plot(history.history['loss'], label='Training Loss', color='red', linewidth=2, linestyle='-.')
plt.plot(history.history['val_loss'], label='Validation Loss', color='blue', linewidth=2, linestyle='-.')

plt.title('Model Loss Curve', weight='bold')
plt.xlabel('Epoch', weight='bold')
plt.ylabel(r'Loss (MSE = $\frac{1}{n} \sum (y_{true} - y_{pred})^2$)', weight='bold')
plt.legend()
plt.show()

# %%
# ---- Predicted vs Actual ----
plt.figure(figsize=(18, 10))
colors = ['blue', 'green', 'purple', 'orange', 'gold']

for i, name in enumerate(output_names):
    plt.subplot(2, 3, i+1)
    plt.scatter(y_actual[:, i], y_pred[:, i], color=colors[i], alpha=0.6)
    plt.plot([min(y_actual[:, i]), max(y_actual[:, i])], [min(y_actual[:, i]), max(y_actual[:, i])], 'r--')
    plt.title(f'({chr(97+i)}) Predicted vs Actual for {name}', weight='bold')
    plt.xlabel('Actual', weight='bold')
    plt.ylabel('Predicted', weight='bold')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Output Names and Labels ----
output_names = ['Indoor (ppm)', 'Outdoor (ppm)', 'Red Zone (miles)', 'Orange Zone (miles)', 'Yellow Zone (miles)']
zone_labels = output_names[2:5]  # Red, Orange, Yellow
stats_labels = ['Min', 'Mean', 'Max']
hatch_patterns = ['///', '...', 'xxx']

# ---- Calculate Summary Stats ----
summary_actual = []
summary_pred = []
input_points = []

X_test_Inv = scaler_X.inverse_transform(X_test)  # Inverse transform for input points

for i in range(2, 5):  # Red, Orange, Yellow
    actual = y_actual[:, i]
    pred = y_pred[:, i]
    summary_actual.append([np.min(actual), np.mean(actual), np.max(actual)])
    summary_pred.append([np.min(pred), np.mean(pred), np.max(pred)])
    # Indices for min, mean (closest), and max
    min_idx = np.argmin(actual)
    max_idx = np.argmax(actual)
    mean_val = np.mean(actual)
    mean_idx = np.argmin(np.abs(actual - mean_val))

    input_points.append([
        X_test_Inv[min_idx],   # Input for min actual
        X_test_Inv[mean_idx],  # Input for mean actual
        X_test_Inv[max_idx]    # Input for max actual
    ])

summary_actual = np.array(summary_actual)
summary_pred = np.array(summary_pred)

# ---- Bar Plot Setup ----
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.5
group_spacing = 1.3
zone_spacing = 1
tick_positions = []
tick_labels = []

zone_colors = [
    ('#ff9999', '#cc0000'),  # Red Zone
    ('#ffcc99', '#ff6600'),  # Orange Zone
    ('#ffff99', '#cccc00'),  # Yellow Zone
]

current_pos = 0

# ---- Draw Bars ----
for zone_idx, (zone, (actual_color, pred_color)) in enumerate(zip(zone_labels, zone_colors)):
    for stat_idx, stat in enumerate(stats_labels):
        actual_val = summary_actual[zone_idx][stat_idx]
        pred_val = summary_pred[zone_idx][stat_idx]

        actual_pos = current_pos
        pred_pos = current_pos + bar_width

        # Bar with hatch patterns
        ax.bar(actual_pos, actual_val, width=bar_width, color=actual_color, edgecolor='black',
               hatch=hatch_patterns[stat_idx], label=f'{stat} - Actual' if zone_idx == 0 else "")
        ax.bar(pred_pos, pred_val, width=bar_width, color=pred_color, edgecolor='black',
               hatch=hatch_patterns[stat_idx], label=f'{stat} - Predicted' if zone_idx == 0 else "")

        # Add tick labels
        tick_positions.append((actual_pos + pred_pos)/2)
        if stat == 'Mean':
            tick_labels.append(f'{stat} \n{zone}')
        else:
            tick_labels.append(f'{stat}')

        current_pos += group_spacing

    current_pos += zone_spacing

# ---- Axis and Legend ----
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=12, weight='bold')
ax.set_ylabel('Distance (miles)', weight='bold')
ax.set_title('Comparison of Predicted vs Actual Threat Zones', fontsize=14, weight='bold')
# Remove color from legend by creating custom legend handles without color
import matplotlib.patches as mpatches
legend_labels = ['Min - Actual', 'Min - Predicted', 'Mean - Actual', 'Mean - Predicted', 'Max - Actual', 'Max - Predicted']
hatches = ['///', '///', '...', '...', 'xxx', 'xxx']
handles = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch=h, label=l)
    for h, l in zip(hatches, legend_labels)
]
ax.legend(handles=handles, ncol=3, loc='upper left', bbox_to_anchor=(0, 1))
plt.tight_layout()
plt.show()

# %%
# ---- Combined Table of Input Points, Actual, Predicted Summary ----

combined_table = []
for zone_idx, zone in enumerate(zone_labels):
    for stat_idx, stat in enumerate(stats_labels):
        inputs = input_points[zone_idx][stat_idx]
        row = {
            'Zone': zone,
            'Stat': stat,
            'Air Temperature (°C)': inputs[0],
            'Wind Speed (m/s)': inputs[1],
            'Hole Diameter (mm)': inputs[2],
            'Actual': summary_actual[zone_idx][stat_idx],
            'Predicted': summary_pred[zone_idx][stat_idx]
        }
        combined_table.append(row)

combined_df = pd.DataFrame(combined_table)
print("\nCombined Input & Output Summary Table:")
print(combined_df)

# Save to Excel
combined_df.to_excel("combined_input_output_summary.xlsx", index=False)
# %%
# ---- Residual Plots (Normalized) ----
plt.figure(figsize=(18, 10))

for i, name in enumerate(output_names):
    residuals = y_actual[:, i] - y_pred[:, i]
    norm_residuals = residuals / (y_actual[:, i] + 1e-8)

    plt.subplot(2, 3, i+1)
    plt.scatter(y_actual[:, i], norm_residuals, color=colors[i], alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'({chr(97+i)}) Normalized Residuals for {name}', weight='bold')
    plt.xlabel('Actual Value', weight='bold')
    plt.ylabel('Normalized Residuals', weight='bold')
    plt.ylim(-1, 1)

plt.tight_layout()
plt.show()
# %%
