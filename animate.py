import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import numpy as np
import argparse
import os

from plot_distance import download_hdf5

font = {'family' : 'serif',
        'size'   : 14}

plt.rc('font', **font)


parser = argparse.ArgumentParser()

# Command line arguments 
# Ex: python animate.py -s TNG300-1
parser.add_argument("-s", "--sim", required=True, type=str, help="Select simulation from IllustrisTNG")


# Ex: python animate.py -s TNG300-1 -a z
parser.add_argument("-a", "--axis", type=str, default="y", help="Select dimension to plot on the y-axis (y or z)")


def dist_anim(sub1, sub2, sim_name, y_axis):
    f, f2 = download_hdf5(sim_name, sub1, sub2, snap_start=99)

    a,b,c = np.flip(f["SubhaloPos"][:,0]/1000),np.flip(f["SubhaloPos"][:,1]/1000),np.flip(f["SubhaloPos"][:,2]/1000)
    a2,b2,c2 = np.flip(f2["SubhaloPos"][:,0]/1000),np.flip(f2["SubhaloPos"][:,1])/1000,np.flip(f2["SubhaloPos"][:,2]/1000)

    y_label = "Y [cMpc/h]"
    gif_fn = f"anim_dist_{sub1}&{sub2}.gif"
    if y_axis == "z":
        b = c
        b2 = c2
        y_label = "Z [cMpc/h]"
        gif_fn = f"anim_dist_z_{sub1}&{sub2}.gif"

    fig,ax = plt.subplots()
    plt.grid()

    ax.set_xlim([min(a) - 20, max(a) + 20])
    ax.set_ylim([min(b) - 20, max(b) + 20])

    ax.set_xlabel("X [cMpc/h]")
    ax.set_ylabel(y_label)

    animated_plot, = ax.plot([],[], 'o',markersize=4, color='red', label="Subhalo " + str(sub1))
    animated_plot2, = ax.plot([],[], 'o',markersize=4, color='blue', label="Subhalo " + str(sub2))

    def update_data(frame):
        animated_plot.set_data([a[frame]], [b[frame]])
        animated_plot2.set_data([a2[frame]], [b2[frame]])
        fig.suptitle(f"Snapshot = {frame}")
        return animated_plot,animated_plot2
    
    if len(a) > len(a2):
        frames = len(a2)
    else:
        frames = len(a)

    anim = animation.FuncAnimation(fig=fig,
                                func=update_data,
                                frames=frames,
                                interval=60,
                                repeat=False)

    fig.legend()
    plt.show()
    anim_dir = f"./{sim_name}/animations/"
    if not os.path.exists(anim_dir):
        os.makedirs(anim_dir)
    fn = anim_dir + gif_fn
    anim.save(fn)
    print(f"Animation saved at {fn}")

if __name__ == '__main__':
    args = parser.parse_args()

    sim_name = args.sim
    y_axis = args.axis

    # Ex: If subhalos 0 and 1 are desired, type the following after the prompt: 0 1
    subhalo_ids = list(input("Which pair of subhalos would you like to plot? Separate by a space: ").split(" "))
    sub1 = subhalo_ids[0]
    sub2 = subhalo_ids[1]

    print("------------------------------------------")
    dist_anim(sub1, sub2, sim_name, y_axis)