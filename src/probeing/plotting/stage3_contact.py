"""Publication-oriented EXP-0008 coupled-vehicle figures."""
from __future__ import annotations
import csv, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

def _rows(path): return list(csv.DictReader(Path(path).open(encoding="utf-8")))
def _save(fig,run,results,name):
    fig.tight_layout()
    for suffix in ("png","pdf"):
        local=run/"figures"/f"{name}.{suffix}"; fig.savefig(local,dpi=220 if suffix=="png" else None,bbox_inches="tight")
        target=results/f"{run.name}__{name}.{suffix}"
        if not target.exists(): shutil.copy2(local,target)
    plt.close(fig)
def _f(row,key): return float(row[key])

def generate_stage3_figures(run_dir: Path, results_dir: Path) -> None:
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,"savefig.facecolor":"white"})
    physics=_rows(run_dir/"physics_metrics.csv"); predictions=_rows(run_dir/"predictions.csv"); extended=_rows(run_dir/"classification_summary_extended.csv")
    fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.set_aspect("equal"); ax.set_xlim(-.42,.22); ax.set_ylim(.72,1.17)
    ax.add_patch(Rectangle((-.2,.93),.18,.1,color="#315b7d",alpha=.8))
    for x,y in [(-.28,1.1),(-.28,.86),(-.08,1.1),(-.08,.86)]: ax.add_patch(Circle((x,y),.035,color="#555"))
    ax.plot([-.11,.17],[.92,.92],lw=5,color="#c98b35"); ax.axvline(.18,color="black",lw=2)
    ax.add_patch(FancyArrowPatch((.17,.92),(.05,.92),arrowstyle="-|>",mutation_scale=18,color="#c43c39")); ax.add_patch(FancyArrowPatch((.17,.92),(.17,1.08),arrowstyle="-|>",mutation_scale=18,color="#6a3d9a"))
    ax.text(-.11,.98,"6-DoF UAV",color="white",ha="center",weight="bold"); ax.text(.01,.875,"rigid offset probe"); ax.text(.19,.94,"moving target",rotation=90); ax.set_title("EXP-0008 offset-contact geometry (world x right, z up)"); ax.axis("off"); _save(fig,run_dir,results_dir,"01_uav_probe_contact_geometry")
    hover=np.load(run_dir/"raw/no_contact_hover.npz"); fig,ax=plt.subplots(figsize=(7.2,3.6))
    for i,l in enumerate("xyz"): ax.plot(hover["time_s"],1e3*(hover["position_world_m"][:,i]-hover["position_world_m"][0,i]),label=l)
    ax.set(xlabel="time (s)",ylabel="position drift (mm)",title="No-contact 10 s hover validation"); ax.legend(ncol=3); _save(fig,run_dir,results_dir,"02_no_contact_hover")
    tr=np.load(run_dir/"raw/no_contact_translation.npz"); fig,ax=plt.subplots(figsize=(7.2,3.8))
    for i,l in enumerate("xyz"): ax.plot(tr["time_s"],tr["position_world_m"][:,i]-tr["position_world_m"][0,i],label=f"{l} actual"); ax.plot(tr["time_s"],tr["desired_position_world_m"][:,i]-tr["position_world_m"][0,i],"--",lw=.8)
    ax.set(xlabel="time (s)",ylabel="position change (m)",title="3-D translational controller validation"); ax.legend(ncol=3); _save(fig,run_dir,results_dir,"03_translation_controller")
    att=np.load(run_dir/"raw/no_contact_attitude.npz"); yaw=np.load(run_dir/"raw/no_contact_yaw.npz"); fig,a=plt.subplots(1,2,figsize=(8,3.5)); a[0].plot(att["time_s"],np.rad2deg(att["euler_xyz_rad"][:,0])); a[0].axhline(5,ls="--",c="k"); a[0].set(title="5° roll step",xlabel="time (s)",ylabel="roll (deg)"); a[1].plot(yaw["time_s"],np.rad2deg(yaw["euler_xyz_rad"][:,2])); a[1].axhline(15,ls="--",c="k"); a[1].set(title="15° yaw step",xlabel="time (s)",ylabel="yaw (deg)"); _save(fig,run_dir,results_dir,"04_attitude_controller")
    cv=np.load(run_dir/"raw/contact_validation.npz"); fig,a=plt.subplots(1,2,figsize=(8.2,3.5)); a[0].plot(cv["press_time_s"],cv["press_force_n"]); a[0].set(title="Stationary press force",xlabel="time (s)",ylabel="force (N)"); a[1].semilogy(cv["press_time_s"],np.maximum(cv["press_total_energy_j"],1e-16)); a[1].set(title="Target + interface energy",xlabel="time (s)",ylabel="energy (J)"); _save(fig,run_dir,results_dir,"05_contact_force_torque_validation")
    stiff=np.load(run_dir/"raw/representative_stiff_probe.npz"); t=stiff["time_s"]; fig,ax=plt.subplots(figsize=(7.5,3.7)); ax.plot(t,stiff["desired_probe_force_n"],label="desired",lw=1.6); ax.plot(t,stiff["realized_contact_force_n"],label="realized",lw=1); ax.axvspan(3,3.5,color="#888",alpha=.15,label="passive"); ax.set(xlabel="time (s)",ylabel="force (N)",title="Desired versus realized probe — stiff target"); ax.legend(); _save(fig,run_dir,results_dir,"06_desired_vs_realized_probe_force")
    mask=t<=3; dt=t[1]-t[0]; freq=np.fft.rfftfreq(mask.sum(),dt); ref=np.abs(np.fft.rfft(stiff["desired_probe_force_n"][mask]-np.mean(stiff["desired_probe_force_n"][mask]))); actual=np.abs(np.fft.rfft(stiff["realized_contact_force_n"][mask]-np.mean(stiff["realized_contact_force_n"][mask]))); fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.semilogy(freq,ref+1e-9,label="desired"); ax.semilogy(freq,actual+1e-9,label="realized"); ax.set_xlim(0,12); ax.axvspan(.5,5,color="#6baed6",alpha=.12); ax.set(xlabel="frequency (Hz)",ylabel="FFT magnitude",title="Delivered spectrum and re-contact harmonics"); ax.legend(); _save(fig,run_dir,results_dir,"07_desired_vs_realized_probe_spectrum")
    fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.plot(t,1e3*stiff["target_displacement_m"]); ax.axvline(3,c="k",ls="--"); ax.set(xlabel="time (s)",ylabel="target displacement (mm)",title="Target response and passive ring-down"); _save(fig,run_dir,results_dir,"08_target_displacement_ringdown")
    fig,ax=plt.subplots(figsize=(7.2,3.6));
    for i,l in enumerate("xyz"): ax.plot(t,1e3*(stiff["vehicle_position_world_m"][:,i]-stiff["vehicle_position_world_m"][0,i]),label=l)
    ax.set(xlabel="time (s)",ylabel="vehicle displacement (mm)",title="UAV motion during probing"); ax.legend(ncol=3); _save(fig,run_dir,results_dir,"09_uav_position_during_probe")
    fig,a=plt.subplots(2,1,figsize=(7.4,5.2),sharex=True)
    for i,l in enumerate(("roll","pitch","yaw")): a[0].plot(t,np.rad2deg(stiff["vehicle_euler_xyz_rad"][:,i]),label=l); a[1].plot(t,stiff["vehicle_angular_velocity_body_rad_s"][:,i],label=l)
    a[0].set(ylabel="attitude (deg)",title="Offset-contact attitude disturbance"); a[0].legend(ncol=3); a[1].set(xlabel="time (s)",ylabel="body rate (rad/s)"); _save(fig,run_dir,results_dir,"10_uav_attitude_and_rates")
    fig,ax=plt.subplots(figsize=(7.2,3.6));
    for i in range(4): ax.plot(t,stiff["rotor_thrust_n"][:,i],label=f"rotor {i+1}")
    ax.set(xlabel="time (s)",ylabel="thrust (N)",title="Individual rotor thrusts during contact"); ax.legend(ncol=4); _save(fig,run_dir,results_dir,"11_rotor_thrusts")
    fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.plot(t,100*stiff["actuator_reserve"],label="minimum reserve"); ax.fill_between(t,0,100*stiff["motor_saturated"].astype(float),alpha=.25,label="saturation"); ax.set(xlabel="time (s)",ylabel="percent",title="Actuator reserve and saturation"); ax.legend(); _save(fig,run_dir,results_dir,"12_actuator_reserve")
    for src,name,title in (("representative_stiff_probe.npz","13_safe_target_example","Representative SAFE target"),("representative_compliant_probe.npz","14_non_safe_target_example","Representative NON-SAFE target")):
        d=np.load(run_dir/"raw"/src); fig,a=plt.subplots(2,1,figsize=(7.4,5),sharex=True); a[0].plot(d["time_s"],d["desired_probe_force_n"],label="desired"); a[0].plot(d["time_s"],d["realized_contact_force_n"],label="realized"); a[0].set(ylabel="force (N)",title=title); a[0].legend(); a[1].plot(d["time_s"],1e3*d["target_displacement_m"]); a[1].axvline(3,ls="--",c="k"); a[1].set(xlabel="time (s)",ylabel="target x (mm)"); _save(fig,run_dir,results_dir,name)
    nominal=[r for r in predictions if r["noise_regime"]=="nominal"]; confusion=np.zeros((2,2),int)
    for r in nominal: confusion[0 if r["actual_safe"]=="True" else 1,0 if r["predicted_safe"]=="True" else 1]+=1
    fig,ax=plt.subplots(figsize=(4.4,3.8)); im=ax.imshow(confusion,cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(confusion[i,j]),ha="center",va="center",fontsize=13)
    ax.set_xticks([0,1],["SAFE","NON-SAFE"]); ax.set_yticks([0,1],["SAFE","NON-SAFE"]); ax.set(xlabel="frozen prediction",ylabel="coupled truth",title="Nominal frozen-policy confusion"); fig.colorbar(im,ax=ax); _save(fig,run_dir,results_dir,"15_frozen_decision_confusion")
    overall=[r for r in extended if r["stratum"]=="target_population"]; x=np.arange(3); rate=np.array([100*_f(r,"false_safe_rate") for r in overall]); upper=np.array([100*_f(r,"false_safe_one_sided95_upper") for r in overall]); fig,ax=plt.subplots(figsize=(6.4,3.7)); ax.bar(x,rate,color="#c44e52"); ax.errorbar(x,rate,yerr=[np.zeros(3),upper-rate],fmt="none",c="k",capsize=4); ax.axhline(2,ls="--",c="k",label="Stage 1 high-noise gate"); ax.set_xticks(x,[r["noise_regime"] for r in overall]); ax.set(ylabel="false-safe rate (%)",title="Coupled false-safe rate (94-target population)"); ax.legend(); _save(fig,run_dir,results_dir,"16_false_safe_results")
    one=[r for r in physics if r["stratum"]=="one_factor"]
    for field,label,name in (("probe_offset_multiplier","probe offset multiplier","17_performance_vs_probe_offset"),("contact_normal_angle_deg","contact-normal angle (deg)","18_performance_vs_contact_angle")):
        s=sorted([r for r in one if r["perturbation"]==field],key=lambda r:_f(r,"perturbation_value")); fig,ax=plt.subplots(figsize=(6.2,3.5)); ax.plot([_f(r,"perturbation_value") for r in s],[_f(r,"probe_rms_tracking_error_n") for r in s],"o-"); ax.set(xlabel=label,ylabel="probe RMS error (N)",title=f"Probe delivery versus {label}"); _save(fig,run_dir,results_dir,name)
    k=np.array([_f(r,"stiffness_n_per_m") for r in physics]); c=np.array([_f(r,"damping_n_s_per_m") for r in physics]); m=np.array([_f(r,"effective_mass_kg") for r in physics]); err=np.array([_f(r,"probe_rms_tracking_error_n") for r in physics]); fig,a=plt.subplots(1,2,figsize=(8.2,3.6)); sc=a[0].scatter(np.sqrt(k/m)/(2*np.pi),err,c=np.log10(c),cmap="viridis",s=24); a[0].set(xlabel="target natural frequency (Hz)",ylabel="probe RMS error (N)",title="Tracking error vs dynamics"); fig.colorbar(sc,ax=a[0],label="log10 damping"); a[1].scatter([_f(r,"contact_loss_count") for r in physics],err,c=k,cmap="plasma",s=24); a[1].set(xlabel="contact-loss events",ylabel="probe RMS error (N)",title="Re-contact degrades delivery"); _save(fig,run_dir,results_dir,"19_probe_tracking_vs_target_dynamics")
    ma=_rows(run_dir/"matlab_v2/matlab_coupled_response.csv"); py=_rows(run_dir/"python_stiff_coupled_response.csv"); mt=np.array([_f(r,"time_s") for r in ma]); mf=np.array([_f(r,"contact_force_n") for r in ma]); pt=np.array([_f(r,"time_s") for r in py]); pf=np.array([_f(r,"contact_force_n") for r in py]); fig,a=plt.subplots(2,1,figsize=(7.4,5),sharex=True); a[0].plot(pt,pf,label="Python"); a[0].plot(mt,mf,"--",label="MATLAB"); a[0].set(ylabel="contact force (N)",title="Independent MATLAB/Python cross-check"); a[0].legend(); a[1].plot(mt,mf-np.interp(mt,pt,pf)); a[1].set(xlabel="time (s)",ylabel="difference (N)"); _save(fig,run_dir,results_dir,"20_python_matlab_simulink_comparison")
    boundary=[r for r in nominal if r["boundary_case"]=="True"]; fig,ax=plt.subplots(figsize=(6.5,3.7)); ax.scatter(np.arange(len(boundary)),[_f(r,"predicted_risk_score") for r in boundary],c=["#4c9f70" if r["actual_safe"]=="True" else "#c44e52" for r in boundary]); ax.axhline(1,ls="--",c="k"); ax.set(xlabel="boundary case",ylabel="frozen risk score",title="Boundary behavior: green SAFE, red NON-SAFE"); _save(fig,run_dir,results_dir,"21_boundary_case_behavior")
    d=np.load(run_dir/"raw/nominal_false_safe_example.npz"); fig,a=plt.subplots(2,1,figsize=(7.4,5),sharex=True); a[0].plot(d["time_s"],d["desired_probe_force_n"],label="desired"); a[0].plot(d["time_s"],d["realized_contact_force_n"],label="realized"); a[0].set(ylabel="force (N)",title="Nominal false-safe probe"); a[0].legend(); a[1].plot(d["time_s"],1e3*d["target_displacement_m"]); a[1].axvline(3,ls="--",c="k"); a[1].set(xlabel="time (s)",ylabel="target x (mm)"); _save(fig,run_dir,results_dir,"22_false_safe_case_detail")
    comp=_rows(run_dir/"reduced_vs_coupled_truth.csv"); direct=np.array([_f(r,"direct_settling_time_s") for r in comp]); coupled=np.array([_f(r,"coupled_settling_time_s") for r in comp]); changed=np.array([r["binary_label_changed"]=="True" for r in comp]); fig,ax=plt.subplots(figsize=(5.2,4.4)); ax.scatter(direct[~changed],coupled[~changed],s=20,alpha=.6,label="label unchanged"); ax.scatter(direct[changed],coupled[changed],s=32,c="#c44e52",label="label changed"); ax.plot([0,3],[0,3],"k--",lw=1); ax.axhline(2,c="#777",ls=":"); ax.set(xlabel="reduced settling time (s)",ylabel="coupled settling time (s)",title="Coupling changes future outcome"); ax.legend(); _save(fig,run_dir,results_dir,"23_reduced_vs_coupled_settling")
