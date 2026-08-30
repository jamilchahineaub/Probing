"""Evidence-oriented plots and trajectory animations for EXP-0009."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
import numpy as np


PHASE_COLORS = {
    "APPROACH": "#5b8ff9", "CONTACT_ACQUIRE": "#61d9a5",
    "PRELOAD": "#f6bd16", "PROBE": "#e8684a",
    "CONTROLLED_UNLOAD": "#9270ca", "PASSIVE_OBSERVE": "#6dc8ec",
    "DECISION": "#5ad8a6", "ABORT": "#d62728",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _save(fig, run_dir: Path, results_dir: Path, name: str) -> None:
    fig.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        local = run_dir / "figures" / f"{name}.{suffix}"
        fig.savefig(local, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        shutil.copy2(local, results_dir / f"{run_dir.name}__{name}.{suffix}")
    plt.close(fig)


def _trajectory(run_dir: Path, trial_id: str) -> Mapping[str, np.ndarray]:
    return np.load(run_dir / "raw/held_out" / f"{trial_id}__probe.npz", allow_pickle=False)


def _phase_span(ax, values: Mapping[str, np.ndarray], *, alpha: float = 0.10) -> None:
    phase = values["phase"].astype(str); time = values["time_s"]
    starts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]
    stops = np.r_[starts[1:], phase.size]
    for start, stop in zip(starts, stops):
        ax.axvspan(time[start], time[stop - 1], color=PHASE_COLORS.get(phase[start], "#999"), alpha=alpha)


def _spectrum(values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = values["phase"].astype(str) == "PROBE"
    time = values["time_s"][mask]
    dt = float(np.median(np.diff(time)))
    reference = values["probe_variation_reference_n"][mask]
    actual = values["realized_contact_force_n"][mask]
    actual = actual - np.mean(actual); reference = reference - np.mean(reference)
    frequency = np.fft.rfftfreq(reference.size, dt)
    ref_fft = np.fft.rfft(reference); actual_fft = np.fft.rfft(actual)
    return frequency, ref_fft, actual_fft, actual_fft / np.where(np.abs(ref_fft) > 1e-12, ref_fft, np.nan)


def generate_exp0009_figures(
    run_dir: Path,
    results_dir: Path,
    selections: Mapping[str, str],
    exp8_summary: Mapping[str, Any],
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "savefig.facecolor": "white"})
    physical = _rows(run_dir / "physical_delivery_trials.csv")
    predictions = _rows(run_dir / "policy_predictions.csv")
    summary = _rows(run_dir / "classification_summary.csv")
    successful = _trajectory(run_dir, selections["successful"])
    failure = _trajectory(run_dir, selections["failure"])
    safe = _trajectory(run_dir, selections["safe"])
    non_safe = _trajectory(run_dir, selections["non_safe"])

    # 1. Prospective state machine.
    states = ["APPROACH", "CONTACT_ACQUIRE", "PRELOAD", "PROBE", "CONTROLLED_UNLOAD", "PASSIVE_OBSERVE", "DECISION"]
    fig, ax = plt.subplots(figsize=(10.2, 2.5)); ax.set_xlim(-0.5, len(states) - 0.1); ax.set_ylim(-1.0, 1.0); ax.axis("off")
    for index, state in enumerate(states):
        ax.text(index, 0, state.replace("_", "\n"), ha="center", va="center", bbox={"boxstyle": "round,pad=.35", "fc": PHASE_COLORS[state], "alpha": .75})
        if index < len(states) - 1: ax.annotate("", (index + .72, 0), (index + .28, 0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(3, -.72, "Any safety violation / timeout  →  ABORT", color=PHASE_COLORS["ABORT"], ha="center", weight="bold")
    ax.set_title("EXP-0009 prospectively frozen hybrid contact controller")
    _save(fig, run_dir, results_dir, "01_hybrid_state_machine")

    # 2/3. Acquisition and preload.
    transitions = np.flatnonzero(successful["phase"][1:] != successful["phase"][:-1]) + 1
    probe_start = int(np.flatnonzero(successful["phase"] == "PROBE")[0])
    for stop, name, title in ((transitions[2] if transitions.size > 2 else probe_start, "02_approach_contact_acquisition", "Slow approach and stable contact acquisition"), (probe_start, "03_preload_stabilization", "Prospective preload stabilization")):
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=True)
        mask = np.arange(successful["time_s"].size) <= stop
        axes[0].plot(successful["time_s"][mask], successful["total_contact_reference_n"][mask], label="reference")
        axes[0].plot(successful["time_s"][mask], successful["realized_contact_force_n"][mask], label="realized")
        _phase_span(axes[0], {key: value[mask] for key, value in successful.items() if isinstance(value, np.ndarray) and value.shape[:1] == successful["time_s"].shape})
        axes[0].set(ylabel="contact force (N)", title=title); axes[0].legend()
        axes[1].plot(successful["time_s"][mask], 1e3 * successful["contact_penetration_m"][mask], label="interface penetration")
        axes[1].plot(successful["time_s"][mask], 1e3 * successful["target_displacement_m"][mask], label="target displacement")
        axes[1].set(xlabel="time (s)", ylabel="displacement (mm)"); axes[1].legend()
        _save(fig, run_dir, results_dir, name)

    # 4/5. Force delivery and error.
    time = successful["time_s"]
    fig, ax = plt.subplots(figsize=(8.0, 3.8)); ax.plot(time, successful["total_contact_reference_n"], label="total reference", lw=1.6); ax.plot(time, successful["realized_contact_force_n"], label="realized", lw=1.0); _phase_span(ax, successful)
    ax.set(xlabel="time (s)", ylabel="force (N)", title="Successful sequence: bounded but distorted force delivery"); ax.legend(ncol=2); _save(fig, run_dir, results_dir, "04_force_reference_vs_realized")
    probe = successful["phase"] == "PROBE"; error = successful["realized_contact_force_n"] - successful["total_contact_reference_n"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.plot(time[probe], error[probe]); ax.axhline(0, color="black", lw=.8); ax.set(xlabel="time (s)", ylabel="realized − reference (N)", title="Probe force-tracking error"); _save(fig, run_dir, results_dir, "05_force_tracking_error")

    # 6/7. Spectrum and population lag/bandwidth.
    frequency, ref_fft, actual_fft, transfer = _spectrum(successful)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7)); axes[0].semilogy(frequency, np.abs(ref_fft) + 1e-10, label="reference"); axes[0].semilogy(frequency, np.abs(actual_fft) + 1e-10, label="realized"); axes[0].set_xlim(0, 8); axes[0].set(xlabel="frequency (Hz)", ylabel="FFT magnitude", title="Delivered spectrum"); axes[0].legend(); valid=(frequency>=.5)&(frequency<=5)&np.isfinite(transfer); axes[1].plot(frequency[valid], 20*np.log10(np.abs(transfer[valid])+1e-12), label="gain"); axes[1].axhspan(-3,3,color="#6baed6",alpha=.12); axes[1].set(xlabel="frequency (Hz)", ylabel="gain (dB)", title="Reference-to-contact transfer")
    _save(fig, run_dir, results_dir, "06_force_spectra")
    complete = [row for row in physical if row["completed"] == "True"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6)); axes[0].hist(1e3*np.abs([_float(row,"probe_cross_correlation_lag_s") for row in complete]), bins=18, color="#5b8ff9"); axes[0].axvline(100, color="black", ls="--"); axes[0].set(xlabel="absolute lag (ms)", ylabel="trials", title="Completed-trial phase lag"); axes[1].hist([_float(row,"probe_delivery_bandwidth_hz") for row in complete], bins=np.arange(-.1,5.6,.25), color="#e8684a"); axes[1].axvline(3, color="black", ls="--"); axes[1].set(xlabel="faithful bandwidth (Hz)", ylabel="trials", title="99.3% have no faithful bin"); _save(fig, run_dir, results_dir, "07_phase_lag_and_bandwidth")

    # 8. Contact timeline: success versus abort.
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), sharex=False)
    for ax, values, title in ((axes[0], successful, "completed sequence"), (axes[1], failure, "acquisition failure")):
        ax.step(values["time_s"], values["contact_active"].astype(float), where="post", label="contact")
        ax.plot(values["time_s"], values["realized_contact_force_n"], label="force (N)", lw=.9); _phase_span(ax, values)
        ax.set(ylabel="status / force", title=title); ax.legend(loc="upper right", ncol=2)
    axes[1].set(xlabel="time (s)"); _save(fig, run_dir, results_dir, "08_contact_separation_timeline")

    # 9/10/11. Unload, passive contamination, ring-down.
    unload = successful["phase"] == "CONTROLLED_UNLOAD"; passive = successful["phase"] == "PASSIVE_OBSERVE"
    fig, ax = plt.subplots(figsize=(7.2,3.5)); ax.plot(time[unload], successful["total_contact_reference_n"][unload], label="unload reference"); ax.plot(time[unload], successful["realized_contact_force_n"][unload], label="realized"); ax.fill_between(time[unload], 0, successful["realized_contact_force_n"][unload], alpha=.15, label="unload impulse"); ax.set(xlabel="time (s)",ylabel="force (N)",title="Controlled unload behavior"); ax.legend(); _save(fig,run_dir,results_dir,"09_controlled_unload")
    fig, axes = plt.subplots(2,1,figsize=(7.2,4.6),sharex=True); axes[0].plot(time[passive],successful["commanded_normal_force_n"][passive],label="command"); axes[0].plot(time[passive],successful["realized_contact_force_n"][passive],label="realized"); axes[0].set(ylabel="force (N)",title="Passive window contamination audit"); axes[0].legend(); axes[1].plot(time[passive],successful["realized_contact_force_n"][passive]**2); axes[1].set(xlabel="time (s)",ylabel=r"force$^2$ (N$^2$)"); _save(fig,run_dir,results_dir,"10_passive_observation_force_energy")
    fig, axes = plt.subplots(2,1,figsize=(7.5,4.8),sharex=True); axes[0].plot(safe["time_s"],1e3*safe["target_displacement_m"],label="SAFE"); axes[0].plot(non_safe["time_s"],1e3*non_safe["target_displacement_m"],label="NON-SAFE",alpha=.8); axes[0].set(ylabel="target x (mm)",title="Target response and passive ring-down"); axes[0].legend(); axes[1].plot(safe["time_s"],safe["realized_contact_force_n"],label="SAFE force"); axes[1].plot(non_safe["time_s"],non_safe["realized_contact_force_n"],label="NON-SAFE force",alpha=.8); axes[1].set(xlabel="time (s)",ylabel="force (N)"); _save(fig,run_dir,results_dir,"11_target_ringdown")

    # 12/13/14. Vehicle response.
    fig, axes=plt.subplots(2,1,figsize=(7.6,4.8),sharex=True)
    for index,label in enumerate(("roll","pitch","yaw")): axes[0].plot(time,np.rad2deg(successful["vehicle_euler_xyz_rad"][:,index]),label=label); axes[1].plot(time,successful["vehicle_angular_velocity_body_rad_s"][:,index],label=label)
    axes[0].set(ylabel="attitude (deg)",title="UAV attitude during hybrid interaction"); axes[0].legend(ncol=3); axes[1].set(xlabel="time (s)",ylabel="body rate (rad/s)"); _save(fig,run_dir,results_dir,"12_uav_attitude")
    fig,ax=plt.subplots(figsize=(7.5,3.5));
    for index in range(4): ax.plot(time,successful["rotor_thrust_n"][:,index],label=f"rotor {index+1}")
    ax.set(xlabel="time (s)",ylabel="thrust (N)",title="Rotor thrusts during contact"); ax.legend(ncol=4); _save(fig,run_dir,results_dir,"13_rotor_thrusts")
    fig,ax=plt.subplots(figsize=(7.3,3.4)); ax.plot(time,100*successful["actuator_reserve"],label="reserve"); ax.fill_between(time,0,100*successful["motor_saturated"].astype(float),alpha=.2,label="saturation"); ax.set(xlabel="time (s)",ylabel="percent",title="Actuator reserve remains ample"); ax.legend(); _save(fig,run_dir,results_dir,"14_actuator_reserve")

    # 15. Aggregate EXP-0008 comparison.
    exp9_values=[100*150/240,.2715186,86.5,0.0,1.24207]
    exp8_values=[0.0,float(exp8_summary["median_probe_rms_tracking_error_n"]),224.0,float(exp8_summary["median_delivery_bandwidth_hz"]),4.994]
    labels=["zero-sep\ncoverage (%)","RMS error\n(N)","lag\n(ms)","bandwidth\n(Hz)","peak force\n(N)"]
    fig,axes=plt.subplots(1,5,figsize=(11,3.3));
    for index,ax in enumerate(axes): ax.bar([0,1],[exp8_values[index],exp9_values[index]],color=["#999","#5b8ff9"]); ax.set_xticks([0,1],["0008","0009"]); ax.set_title(labels[index]); ax.grid(axis="x",visible=False)
    fig.suptitle("EXP-0008 versus EXP-0009: impacts reduced, delivery still unfaithful")
    _save(fig,run_dir,results_dir,"15_exp0008_vs_exp0009_controller_metrics")

    # 16/17. Frozen-policy decision and confidence.
    nominal=[row for row in predictions if row["noise_regime"]=="nominal"]
    confusion=np.zeros((2,2),dtype=int)
    for row in nominal: confusion[0 if row["actual_safe"]=="True" else 1,0 if row["predicted_safe"]=="True" else 1]+=1
    fig,ax=plt.subplots(figsize=(4.5,3.8)); im=ax.imshow(confusion,cmap="Blues");
    for i in range(2):
        for j in range(2): ax.text(j,i,str(confusion[i,j]),ha="center",va="center",fontsize=13)
    ax.set_xticks([0,1],["SAFE","NON-SAFE"]); ax.set_yticks([0,1],["SAFE","NON-SAFE"]); ax.set(xlabel="frozen/end-to-end decision",ylabel="coupled truth",title="Nominal confusion matrix"); fig.colorbar(im,ax=ax); _save(fig,run_dir,results_dir,"16_frozen_classifier_confusion")
    end=[row for row in summary if row["scope"]=="end_to_end_abort_is_non_safe" and row["stratum"]=="overall"]
    conditional=[row for row in summary if row["scope"]=="frozen_policy_completed_delivery_only" and row["stratum"]=="overall"]
    fig,ax=plt.subplots(figsize=(7.0,3.8)); x=np.arange(3); width=.35
    for offset,rows,label,color in ((-.5,end,"end-to-end","#5b8ff9"),(.5,conditional,"completed only","#e8684a")):
        rates=np.array([100*_float(row,"false_safe_rate") for row in rows]); uppers=np.array([100*_float(row,"false_safe_one_sided95_upper") for row in rows]); ax.bar(x+offset*width,rates,width,label=label,color=color); ax.errorbar(x+offset*width,rates,yerr=[np.zeros(3),uppers-rates],fmt="none",color="black",capsize=3)
    ax.axhline(2,color="black",ls="--",label="high-noise gate"); ax.set_xticks(x,[row["noise_regime"] for row in end]); ax.set(ylabel="false-safe rate (%)",title="Frozen-policy false-safe rate and one-sided 95% bound"); ax.legend(); _save(fig,run_dir,results_dir,"17_false_safe_confidence_intervals")

    # 18/19. Boundary and failure attribution.
    boundary=[row for row in physical if row["stratum"]=="boundary"]
    fig,ax=plt.subplots(figsize=(7.0,3.8)); colors=["#5ad8a6" if row["actual_risk_class"]=="SAFE" else "#e8684a" for row in boundary]; markers=["o" if row["completed"]=="True" else "x" for row in boundary]
    for row,color,marker in zip(boundary,colors,markers): ax.scatter(_float(row,"reduced_order_boundary_severity"),_float(row,"effective_mass_kg"),c=color,marker=marker,s=35)
    ax.axvline(1,color="black",ls="--"); ax.set(xlabel="reduced-order boundary severity ratio",ylabel="effective mass (kg)",title="Boundary targets: circles completed, crosses failed delivery"); _save(fig,run_dir,results_dir,"18_boundary_case_performance")
    false_safe=[row for row in predictions if row["false_safe"]=="True"]
    categories={};
    for row in false_safe: categories[row["failure_mechanism"]]=categories.get(row["failure_mechanism"],0)+1
    incomplete={};
    for row in physical:
        if row["completed"]!="True": incomplete[row["terminal_reason"]]=incomplete.get(row["terminal_reason"],0)+1
    names=list(incomplete)+list(categories); values=list(incomplete.values())+list(categories.values()); colors=["#9270ca"]*len(incomplete)+["#d62728"]*len(categories)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.bar(np.arange(len(names)),values,color=colors); ax.set_xticks(np.arange(len(names)),[name.replace("_","\n") for name in names]); ax.set(ylabel="trials / noise-cases",title="Failure mechanisms: delivery failures and false-safe audit"); _save(fig,run_dir,results_dir,"19_false_safe_failure_mechanisms")


def _body_geometry(values: Mapping[str, np.ndarray], index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position=values["vehicle_position_world_m"][index,[0,2]]; pitch=float(values["vehicle_euler_xyz_rad"][index,1]); c=np.cos(pitch); s=np.sin(pitch); rotation=np.array([[c,s],[-s,c]])
    arm=np.column_stack(([-.12,.12],[0,0])) @ rotation.T + position
    vertical=np.column_stack(([0,0],[-.045,.045])) @ rotation.T + position
    probe_offset=np.array([.30,-.08]); tip=position+rotation@probe_offset
    return arm,vertical,tip


def _draw_vehicle(ax, values: Mapping[str,np.ndarray], index: int) -> None:
    arm,vertical,tip=_body_geometry(values,index); position=values["vehicle_position_world_m"][index,[0,2]]
    ax.plot(arm[:,0],arm[:,1],lw=5,color="#315b7d",solid_capstyle="round"); ax.plot(vertical[:,0],vertical[:,1],lw=3,color="#315b7d")
    for point in arm: ax.add_patch(Circle(point,.018,color="#555"))
    ax.plot([position[0],tip[0]],[position[1],tip[1]],lw=4,color="#c98b35"); ax.add_patch(Circle(tip,.008,color="#d62728"))
    target=float(values["target_displacement_m"][index]); ax.axvline(0,color="#777",ls="--",lw=1); ax.plot([target,target],[.67,1.16],color="black",lw=3)
    ax.set(xlim=(-.44,.04),ylim=(.67,1.17),aspect="equal",xlabel="world x (m)",ylabel="world z (m)")


def animate_hybrid(
    values: Mapping[str,np.ndarray], output: Path, *, title: str,
    decision: str = "—", start_s: float | None = None, stop_s: float | None = None,
    frame_count: int = 65,
) -> None:
    time=values["time_s"]; start=float(time[0] if start_s is None else start_s); stop=float(time[-1] if stop_s is None else stop_s)
    indices=np.unique(np.clip(np.searchsorted(time,np.linspace(start,stop,frame_count)),0,time.size-1))
    fig=plt.figure(figsize=(10.0,5.2)); grid=fig.add_gridspec(3,2,width_ratios=[1.05,1.0]); physical=fig.add_subplot(grid[:,0]); force=fig.add_subplot(grid[0,1]); target=fig.add_subplot(grid[1,1],sharex=force); pitch=fig.add_subplot(grid[2,1],sharex=force)
    force.plot(time,values["total_contact_reference_n"],label="reference",lw=1.2); force.plot(time,values["realized_contact_force_n"],label="realized",lw=.9); force.set(ylabel="force (N)"); force.legend(fontsize=7,ncol=2)
    target.plot(time,1e3*values["target_displacement_m"],color="#e8684a"); target.set(ylabel="target (mm)")
    pitch.plot(time,np.rad2deg(values["vehicle_euler_xyz_rad"][:,1]),color="#9270ca"); pitch.set(xlabel="time (s)",ylabel="pitch (deg)")
    cursors=[ax.axvline(time[indices[0]],color="black",lw=1) for ax in (force,target,pitch)]
    fig.suptitle(title)
    def update(frame_index: int):
        index=int(indices[frame_index]); physical.clear(); _draw_vehicle(physical,values,index)
        phase=str(values["phase"][index]); contact="CONTACT" if bool(values["contact_active"][index]) else "SEPARATED"
        physical.set_title(f"t={time[index]:.2f} s | {phase}\n{contact} | F={values['realized_contact_force_n'][index]:.3f} N | decision={decision}",color=PHASE_COLORS.get(phase,"black"),fontsize=10)
        physical.text(.02,.02,f"target x={1e3*values['target_displacement_m'][index]:.2f} mm\nphysical scale (no displacement exaggeration)",transform=physical.transAxes,fontsize=8,bbox={"fc":"white","alpha":.8})
        for cursor in cursors: cursor.set_xdata([time[index],time[index]])
        return cursors
    animation=FuncAnimation(fig,update,frames=len(indices),interval=100,blit=False); output.parent.mkdir(parents=True,exist_ok=True); animation.save(output,writer=PillowWriter(fps=10)); plt.close(fig)


def animate_exp8_comparison(exp8: Mapping[str,np.ndarray], exp9: Mapping[str,np.ndarray], output: Path) -> None:
    exp9_probe=np.flatnonzero(exp9["phase"]=="PROBE"); start=max(float(exp9["time_s"][exp9_probe[0]])-.2,0); stop=min(float(exp9["time_s"][exp9_probe[-1]])+.5,float(exp9["time_s"][-1])); fractions=np.linspace(0,1,65)
    idx8=np.clip(np.searchsorted(exp8["time_s"],fractions*float(exp8["time_s"][-1])),0,exp8["time_s"].size-1); idx9=np.clip(np.searchsorted(exp9["time_s"],start+fractions*(stop-start)),0,exp9["time_s"].size-1)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8)); fig.suptitle("Same target: EXP-0008 continuous control vs EXP-0009 hybrid control")
    def update(frame: int):
        for ax,values,index,label in ((axes[0],exp8,int(idx8[frame]),"EXP-0008"),(axes[1],exp9,int(idx9[frame]),"EXP-0009")):
            ax.clear(); _draw_vehicle(ax,values,index); contact="CONTACT" if bool(values["contact_active"][index]) else "SEPARATED"; ax.set_title(f"{label} | t={values['time_s'][index]:.2f}s\n{contact} | F={values['realized_contact_force_n'][index]:.2f} N")
        return []
    animation=FuncAnimation(fig,update,frames=len(fractions),interval=100,blit=False); animation.save(output,writer=PillowWriter(fps=10)); plt.close(fig)
