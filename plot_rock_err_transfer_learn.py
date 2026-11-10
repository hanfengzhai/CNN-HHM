import os, argparse, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Patch
from util_visualization import plotting_params

plt.rcParams.update(plotting_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE and rock slices")
    parser.add_argument('--fig-dir', default='figures')
    parser.add_argument('--base-dir', default='/path/to/digital_rock/rock_images_segmented')
    parser.add_argument('--size', type=int, default=1200)
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    # RMSE bar‐chart
    keys = ['Stiffness Tensor','Elastic Moduli',
            'HS Bound Factor\n(Ours)','HS Bound Factor\n(Benchmark 1)']
    sample_sizes = [300,600]
    models = ['CG','B2','FB1','FB2']
    colors = ['darkblue','darkred','darkgreen','darkorange']
    rmse_CG = {300:{'K':[.25,.06,.22,.46],'G':[.30,.07,.23,.61]},
               600:{'K':[.25,.04,.21,.49],'G':[.29,.05,.22,.63]}}
    rmse_B2 = {300:{'K':[.21,.07,.11,.38],'G':[.24,.08,.13,.52]},
               600:{'K':[.20,.03,.08,.35],'G':[.24,.04,.09,.52]}}
    rmse_FB1= {300:{'K':[.17,.07,.11,.51],'G':[.22,.09,.12,.74]},
               600:{'K':[.17,.05,.09,.51],'G':[.22,.08,.10,.75]}}
    rmse_FB2= {300:{'K':[.03,.11,.15,.25],'G':[.07,.12,.20,.40]},
               600:{'K':[.02,.11,.15,.24],'G':[.06,.12,.20,.39]}}
    R2_CG = {300:{'K':[.977,.977,.993,.922],'G':[.982,.998,.995,.912]},
               600:{'K':[.837,.997,.971,.481],'G':[.865,.997,.982,.343]}}
    R2_B2 = {300:{'K':[.934,.996,.948,.753],'G':[.942,.997,.967,.761]},
               600:{'K':[.673,.991,.750,-.306],'G':[..676,.990,.816,-.527]}}
    R2_FB1= {300:{'K':[.966,.995,.987,.696],'G':[.972,.996,.991,.691]},
               600:{'K':[.720,.971,.919,-1.608],'G':[.744,.969,.947,-1.924]}}
    R2_FB2= {300:{'K':[.998,.980,.963,.892],'G':[.997,990,.970,878]},
               600:{'K':[.990,.752,.543,.-.204],'G':[.964,.876,.643,-.419]}}
    rmse_data = dict(zip(models,[rmse_CG,rmse_B2,rmse_FB1,rmse_FB2]))

    x = np.arange(len(keys))
    bar_w = 0.8/len(models)
    offsets = (np.arange(len(models))-(len(models)-1)/2)*bar_w

    fig, axes = plt.subplots(2,2,figsize=(10,9),sharey=True)
    labels = ['(a)','(b)','(c)','(d)']
    for i,metric in enumerate(['K','G']):
        for j,size in enumerate(sample_sizes):
            ax = axes[i,j]
            ax.text(0.02,0.95,labels[i*2+j],transform=ax.transAxes,fontsize=20,va='top')
            for k,model in enumerate(models):
                ax.bar(x+offsets[k], rmse_data[model][size][metric],
                       width=bar_w, color=colors[k], edgecolor='black',
                       label=model if (i,j)==(0,0) else "")
            ax.set_xticks(x); ax.set_xticklabels(keys,rotation=45,ha='right')
            ax.text(0.5,0.9,rf"Rock size: ${size}\times{size}\times{size}$",
                    transform=ax.transAxes,ha='center',fontsize=15)
            if j==0: ax.set_ylabel(rf"RMSE for ${metric}$ [GPa]")
            if i==0: ax.set_xlabel(''); ax.set_xticklabels([])
    handles = [Patch(color=colors[m],label=models[m]) for m in range(len(models))]
    fig.legend(handles,models,loc='upper center',bbox_to_anchor=(0.15,0.9))
    plt.tight_layout()
    out1 = os.path.join(args.fig_dir,'rock_err_transfer_learn.png')
    plt.savefig(out1,dpi=300); print(f"---> saved figure: {out1}")

    # load volumes & extract middle z‐slice
    samples = ['B1','B2','CG','FB1','FB2']
    size = args.size; shape = (size,)*3; dtype = np.uint8
    mid = size//2; slices = []
    for name in tqdm(samples,desc="Loading slices"):
        path = os.path.join(args.base_dir,name,f'size_{size}',f'Rock_{size}.raw')
        vol = np.fromfile(path,dtype=dtype).reshape(shape)
        slices.append(vol[:,:,mid])

    fig,axes = plt.subplots(1,len(samples),figsize=(5,27),constrained_layout=True)
    for ax,slc,name in zip(axes,slices,samples):
        ax.imshow(slc,cmap='gray',origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_edgecolor('darkred'); spine.set_linewidth(1)
        ax.text(0.5,-0.05,name,transform=ax.transAxes,ha='center',va='top')
    out2 = os.path.join(args.fig_dir,'rock_slices.png')
    plt.savefig(out2,dpi=300,bbox_inches='tight'); print(f"---> saved figure: {out2}")