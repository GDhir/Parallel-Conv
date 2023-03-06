import os
import subprocess
import re
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots()

cmd = list()

cmd.append("./cnn-gpu 1 3 64 112 112 3 3 2 2")
cmd.append("./cnn-gpu 1 832 128 7 7 1 1 1 1")
cmd.append("./cnn-gpu 128 3 64 112 112 3 3 2 2")
cmd.append("./cnn-gpu 128 832 128 7 7 1 1 1 1")

confs = ["conf1", "conf2", "conf3", "conf4"]

numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | \
                (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'

def runallConfs(root_dir, outfile, compile_cmd, cmd):

    build_dir = root_dir + "/build"

    open(outfile, 'w').close()
    compile_cmdlist = compile_cmd.split(" ")
    subprocess.run(compile_cmdlist, cwd=build_dir)

    for item in cmd:

        cmdlist = item.split(" ")
        subprocess.run(cmdlist, stdout=open(outfile, 'a'),
                       stderr=open(outfile, 'a'), cwd=build_dir)


def plotallConfs(root_dir, outfile):

    build_dir = root_dir + "/build"

    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    confs = ["conf1", "conf2", "conf3", "conf4"]
    vals = []

    with open(outfile, 'r') as fval:

        for idx, lineval in enumerate(fval.readlines()):

            if idx % 2 == 1:
                vals.append(float(rx.findall(lineval)[1]))

    print(vals)
    print(confs)
    bars = ax.bar(confs, vals, width=0.2)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .01, round(yval, 2))

    plt.xlabel("Configuration")
    plt.ylabel("Performance (ms)")
    plt.savefig(build_dir + "/general.png")
    plt.show()


def blockSizeBench(root_dir):

    linenumber = 0

    driverfile = root_dir + "/driver.cu"
    build_dir = root_dir + "/build"
    fullfile = []

    bsizes = [32, 64, 128, 256, 512, 1024]
    allvals = []

    for bsize in bsizes:

        with open(driverfile, "r") as fval:

            fullfile = list(fval.readlines())

            for idx, line in enumerate(fullfile):

                if re.match("#define BLOCK_SIZE", line):
                    linenumber = idx

        newlistval = "#define BLOCK_SIZE " + str(bsize) + "\n"

        newfullfile = fullfile[:linenumber] + \
            [newlistval] + fullfile[linenumber + 1:]

        with open(driverfile, "w") as fval:

            fval.writelines(newfullfile)

        outfile = build_dir + "/out_" + str(bsize) + ".txt"

        runallConfs(root_dir, outfile, "make", cmd)

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    plt.figure()

    allvals = np.array(allvals)

    for idx, conf in enumerate(confs):

        plt.plot(bsizes, allvals[:, idx], "-o", label=conf)

    plt.xlabel("Block Size")
    plt.ylabel("Performance (ms)")
    plt.legend()
    plt.savefig("Block_Size_Benchmark128.png")


def kunrollBench(root_dir):

    kfactors = [2, 4, 8, 16]
    build_dir = root_dir + "/build"

    allvals = []

    for kfactor in kfactors:

        foldername = root_dir + "/unroll_k" + str(kfactor)

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/outk" + str(kfactor) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd[:2])

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    plt.figure()

    allvals = np.array(allvals)

    for idx, conf in enumerate(confs[:2]):

        plt.plot(kfactors, allvals[:, idx], "-o", label=conf)

    plt.xlabel("K loop Unroll Factors")
    plt.ylabel("Performance (ms)")
    plt.legend()
    plt.savefig("KLoopUnroll_Benchmark1_bsize=128.png")


def cunrollBench(root_dir):

    cfactors = [0, 2, 4, 8, 16]
    build_dir = root_dir + "/build"

    allvals = []

    for cfactor in cfactors:

        foldername = root_dir + "/unroll_c" + str(cfactor)

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/outc" + str(cfactor) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd[:2])

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    plt.figure()

    allvals = np.array(allvals)

    for idx, conf in enumerate(confs[:2]):

        plt.plot(cfactors, allvals[:, idx], "-o", label=conf)

    plt.xlabel("C loop Unroll Factors")
    plt.ylabel("Performance (ms)")
    plt.legend()
    plt.savefig("CLoopUnroll_Benchmark1.png")


def generalBench(root_dir):

    build_dir = root_dir + "/build"

    outfile = build_dir + "/out.txt"

    runallConfs(root_dir, outfile, "make", cmd)


def rscpermute(root_dir):

    build_dir = root_dir + "/build"

    outfile = build_dir + "/out.txt"

    foldernames = [root_dir + "/unroll_k8", root_dir + "/unroll_k8_rsc"]
    allvals = []

    for idx, foldername in enumerate(foldernames):

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/outpermute" + str(idx) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd)

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    permutes = ["crs", "rsc"]
    x = np.arange(len(confs))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(constrained_layout=True)

    allvals = np.array(allvals)

    for idx, permute in enumerate(permutes):
        offset = width * multiplier
        rects = ax.bar(x + offset, allvals[idx, :], width, label=permute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Performance (ms)')
    ax.set_xticks(x + width, confs)

    plt.savefig(build_dir + "rsc_vs_crs.png")


def nk_vs_k(root_dir):

    build_dir = root_dir + "/build"

    outfile = build_dir + "/out.txt"

    foldernames = [root_dir + "/unroll_nk", root_dir + "/unroll_k8_rsc"]
    allvals = []

    for idx, foldername in enumerate(foldernames):

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/outpermute" + str(idx) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd[2:])

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    unrollconfs = ["nk", "k"]
    x = np.arange(len(confs[2:]))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(constrained_layout=True)

    allvals = np.array(allvals)

    for idx, unrollconf in enumerate(unrollconfs):
        offset = width * multiplier
        rects = ax.bar(x + offset, allvals[idx, :], width, label=unrollconf)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Performance (ms)')
    ax.set_xticks(x + width, confs[2:])

    plt.savefig(build_dir + "nk_vs_k.png")


def Plain_vs_Reorder(root_dir):

    build_dir = root_dir + "/build"

    outfile = build_dir + "/out.txt"

    foldernames = [root_dir + "/Plain", root_dir + "/Reorder"]
    allvals = []

    for idx, foldername in enumerate(foldernames):

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/reorder" + str(idx) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd)

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    unrollconfs = ["Plain", "Reorder"]
    x = np.arange(len(confs))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(constrained_layout=True)

    allvals = np.array(allvals)

    for idx, unrollconf in enumerate(unrollconfs):
        offset = width * multiplier
        rects = ax.bar(x + offset, allvals[idx, :], width, label=unrollconf)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Performance (ms)')
    ax.set_xticks(x + width, confs)

    plt.savefig(build_dir + "Plain_vs_Reorder.png")


def SM_vs_Plain(root_dir):

    build_dir = root_dir + "/build"

    outfile = build_dir + "/out.txt"

    foldernames = [root_dir + "/Reorder", root_dir + "/SM"]
    allvals = []

    for idx, foldername in enumerate(foldernames):

        fname = foldername + "/cnn.assign.cu"

        compile_cmd = "nvcc -O3 -o cnn-gpu " + fname
        outfile = build_dir + "/SM" + str(idx) + ".txt"

        runallConfs(root_dir, outfile, compile_cmd, cmd[2:])

        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        vals = []

        with open(outfile, 'r') as fval:

            for idx, lineval in enumerate(fval.readlines()):

                if idx % 2 == 1:
                    vals.append(float(rx.findall(lineval)[1]))

        allvals.append(vals)

    unrollconfs = ["Plain", "SM"]
    x = np.arange(len(confs[2:]))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(constrained_layout=True)

    allvals = np.array(allvals)

    for idx, unrollconf in enumerate(unrollconfs):
        offset = width * multiplier
        rects = ax.bar(x + offset, allvals[idx, :], width, label=unrollconf)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Performance (ms)')
    ax.set_xticks(x + width, confs[2:])

    plt.savefig(build_dir + "SM_vs_Plain.png")


if __name__ == "__main__":

    root_dir = "/uufs/chpc.utah.edu/common/home/u1444601/CS6235/HW4"
    build_dir = root_dir + "/build"
    outfile = build_dir + "/out.txt"
    # runallConfs( root_dir )
    # plotallConfs( root_dir, outfile )
    # blockSizeBench(root_dir)
    generalBench(root_dir)
    # kunrollBench(root_dir)
    # cunrollBench(root_dir)
    # rscpermute(root_dir)
    # nk_vs_k(root_dir)
    # Plain_vs_Reorder(root_dir)
    # SM_vs_Plain(root_dir)
