"""
Visualize radar frame results with depth perception and accurate doppler radial lines:
- Scatter of radar points colored by predicted class, marker size shaded by range (depth)
- Continuous doppler velocity lines drawn radially, colored by velocity magnitude
- Vehicle silhouette at bottom center
- RL focus wedge
- Per-point class F1-score and other metrics bar charts (aggregated over simulation)
- 3D inset showing range vs depth
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import hydra
from omegaconf import DictConfig
from datasets.radar_scenes import RadarScenesDataset

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    root = get_original_cwd()

    print("--- Generating Visualizations ---")

    log_path = os.path.join(root, cfg.sim.output_dir, cfg.sim.log_filename)
    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        full_log = json.load(f)

    logs = full_log.get('frames', [])
    overall_point_metrics = full_log.get('point_metrics_overall', {})
    per_class_point_metrics = full_log.get('point_metrics_per_class', {})

    if not logs:
        print("No frame data found in the simulation log.")
        return

    ds = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes
    )
    out_dir = os.path.join(root, cfg.sim.output_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)

    classes_for_plot = list(range(cfg.scene.num_classes))
    cmap_classes = plt.cm.tab10(np.linspace(0,1,len(classes_for_plot)))

    xlim_plot = 70
    ylim_plot_min = -10
    ylim_plot_max = 120

    for entry in logs:
        if not isinstance(entry, dict):
            continue
        idx = int(entry.get('frame_index', 0))
        ts  = entry.get('timestamp', None)
        rl_az, rl_span = entry.get('rl_action', [0,0])
        
        pred_point_labels_logged = np.array(entry.get('pred_point_labels', []))
        coords_logged = np.array(entry.get('coords', []))
        velocities_logged = np.array(entry.get('velocities', []))
        num_points_logged = entry.get('num_points', 0)

        if num_points_logged == 0 or len(pred_point_labels_logged) == 0:
            print(f"Skipping frame {idx}: No points or predicted labels found.")
            continue

        x_cartesian = coords_logged[:, 0]
        y_cartesian = coords_logged[:, 1]
        
        rng = np.sqrt(x_cartesian**2 + y_cartesian**2)
        ang = np.arctan2(x_cartesian, y_cartesian)
        dop = velocities_logged[:, 0]

        fig = plt.figure(figsize=(14,10), facecolor='white')
        ax = fig.add_subplot(2,2,1)

        sizes = np.clip(200 / (rng + 1), 5, 50)
        alphas = np.clip(1 / (1 + rng / 40), 0.2, 1)
        for c, col in zip(classes_for_plot, cmap_classes):
            mask = (pred_point_labels_logged == c)
            if mask.any():
                ax.scatter(x_cartesian[mask], y_cartesian[mask], s=sizes[mask], c=[col], alpha=alphas[mask],
                           label=f'Class {c}', edgecolors='k', linewidths=0.3)

        if len(dop) > 0:
            norm_dop = plt.Normalize(dop.min(), dop.max())
            for xi, yi, ri, vi in zip(x_cartesian, y_cartesian, rng, dop):
                if ri > 1e-6:
                    ux, uy = xi / ri, yi / ri
                else:
                    ux, uy = 0, 1
                length_factor = 0.1
                line_length = np.clip(abs(vi) * length_factor, 0.5, 10)
                ax.plot([xi, xi + ux * line_length], [yi, yi + uy * line_length], lw=1.2,
                        color=plt.cm.plasma(norm_dop(vi)), alpha=0.8)
            sm = plt.cm.ScalarMappable(norm=norm_dop, cmap='plasma')
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Doppler Velocity (m/s)')

        veh_w, veh_h = 2.0, 5.0
        veh_bottom_left_x = -veh_w / 2
        veh_bottom_left_y = -veh_h
        veh = Rectangle((veh_bottom_left_x, veh_bottom_left_y), veh_w, veh_h, color='black', zorder=5)
        ax.add_patch(veh)
        
        wheel_radius = 0.5
        ax.add_patch(Circle((veh_bottom_left_x + wheel_radius, veh_bottom_left_y + wheel_radius), wheel_radius, color='dimgray', zorder=6))
        ax.add_patch(Circle((veh_bottom_left_x + veh_w - wheel_radius, veh_bottom_left_y + wheel_radius), wheel_radius, color='dimgray', zorder=6))
        ax.add_patch(Circle((veh_bottom_left_x + wheel_radius, veh_bottom_left_y + veh_h - wheel_radius), wheel_radius, color='dimgray', zorder=6))
        ax.add_patch(Circle((veh_bottom_left_x + veh_w - wheel_radius, veh_bottom_left_y + veh_h - wheel_radius), wheel_radius, color='dimgray', zorder=6))

        range_max_for_wedge = 100
        wedge = Wedge((0,0), range_max_for_wedge, np.degrees(rl_az-rl_span/2),
                      np.degrees(rl_az+rl_span/2), color='red', alpha=0.2)
        ax.add_patch(wedge)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-xlim_plot, xlim_plot)
        ax.set_ylim(ylim_plot_min, ylim_plot_max)
        
        ax.set_xlabel('Lateral (m)'); ax.set_ylabel('Forward (m)')
        ax.set_title(f'Frame {idx} @ {ts} (Predicted Point Labels)', fontsize=14)
        ax.legend(loc='upper right', ncol=2, fontsize='small', framealpha=0.6)
        ax.grid(True, linestyle=':', alpha=0.6)

        ax3 = fig.add_subplot(2,2,2, projection='3d')
        ax3.scatter(x_cartesian, y_cartesian, rng, c=rng, cmap='viridis', s=15, alpha=0.7)
        ax3.set_xlabel('Lateral'); ax3.set_ylabel('Forward'); ax3.set_zlabel('Range (m)')
        ax3.set_title('3D Depth View')

        ax2 = fig.add_subplot(2,2,3)
        
        f1_vals_to_plot = [0.0] * len(classes_for_plot)
        for c in classes_for_plot:
            metric_data = per_class_point_metrics.get(str(c), {})
            f1_vals_to_plot[c] = metric_data.get('f1', 0.0)

        ax2.bar(classes_for_plot, f1_vals_to_plot, color=[cmap_classes[c] for c in classes_for_plot])
        ax2.set_xticks(classes_for_plot); ax2.set_xticklabels([f'C{c}' for c in classes_for_plot])
        ax2.set_ylim(0,1); ax2.set_title('Overall Per-Class F1-Score (Simulation)')
        ax2.set_ylabel('F1-Score')
        ax2.grid(True, linestyle=':', alpha=0.6, axis='y')

        ax4 = fig.add_subplot(2,2,4)
        ax4.axis('off')
        
        overall_acc = overall_point_metrics.get('accuracy', 0.0)
        overall_macro_f1 = overall_point_metrics.get('f1', 0.0)
        
        text_content = (f"Simulation Summary:\n"
                        f"Overall Accuracy: {overall_acc:.3f}\n"
                        f"Macro F1-Score: {overall_macro_f1:.3f}\n\n"
                        f"Details in {os.path.basename(cfg.sim.log_filename)}")
        
        ax4.text(0.1, 0.9, text_content, transform=ax4.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax4.set_title('Simulation Metrics Overview')

        plt.tight_layout()
        out = os.path.join(out_dir, f'frame_{idx}.png')
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved: {out}")

    print("--- Visualizations Generated ---")

if __name__=='__main__':
    main()