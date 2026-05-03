import sys
sys.path.append('../')

from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input: 5-channel events voxel grid
    to_Conv(
        name="events",
        s_filer="H×W",
        n_filer=5,
        offset="(0,0,0)",
        to="(0,0,0)",
        width=3,
        height=40,
        depth=40,
        caption="Events"
    ),

    # Enc1 + Pool1
    to_ConvConvRelu(
        name='enc1',
        s_filer="H×W",
        n_filer=(32, 32),
        offset="(1.5,0,0)",
        to="(events-east)",
        width=(3, 3),
        height=40,
        depth=40,
        caption="Enc1"
    ),

    to_Pool(
        name="pool1",
        offset="(0,0,0)",
        to="(enc1-east)",
        width=1,
        height=32,
        depth=32,
        opacity=0.5,
        caption=""
    ),

    # Enc2 + Pool2
    to_ConvConvRelu(
        name='enc2',
        s_filer="H/2×W/2",
        n_filer=(64, 64),
        offset="(1.8,0,0)",
        to="(pool1-east)",
        width=(3, 3),
        height=32,
        depth=32,
        caption="Enc2"
    ),

    to_Pool(
        name="pool2",
        offset="(0,0,0)",
        to="(enc2-east)",
        width=1,
        height=24,
        depth=24,
        opacity=0.5,
        caption=""
    ),

    # Bottleneck
    to_ConvConvRelu(
        name='bottleneck',
        s_filer="H/4×W/4",
        n_filer=(128, 128),
        offset="(2.0,0,0)",
        to="(pool2-east)",
        width=(4.5, 4.5),
        height=24,
        depth=24,
        caption="Bottleneck"
    ),

    # Main arrows
    to_connection("events", "enc1"),
    to_connection("pool1", "enc2"),
    to_connection("pool2", "bottleneck"),

    # Up2 + Dec2
    to_UnPool(
        name="up2",
        offset="(2.0,0,0)",
        to="(bottleneck-east)",
        width=1,
        height=32,
        depth=32,
        opacity=0.5,
        caption=""
    ),

    to_ConvConvRelu(
        name='dec2',
        s_filer="H/2×W/2",
        n_filer=(64, 64),
        offset="(0,0,0)",
        to="(up2-east)",
        width=(3, 3),
        height=32,
        depth=32,
        caption="Dec2"
    ),

    to_connection("bottleneck", "up2"),
    to_skip(of='enc2', to='dec2', pos=1.25),

    # Up1 + Dec1
    to_UnPool(
        name="up1",
        offset="(2.0,0,0)",
        to="(dec2-east)",
        width=1,
        height=40,
        depth=40,
        opacity=0.5,
        caption=""
    ),

    to_ConvConvRelu(
        name='dec1',
        s_filer="H×W",
        n_filer=(32, 32),
        offset="(0,0,0)",
        to="(up1-east)",
        width=(3, 3),
        height=40,
        depth=40,
        caption="Dec1"
    ),

    to_connection("dec2", "up1"),
    to_skip(of='enc1', to='dec1', pos=1.25),

    # Output
    to_Conv(
        name="out_conv",
        s_filer="H×W",
        n_filer=3,
        offset="(1.2,0,0)",
        to="(dec1-east)",
        width=3,
        height=40,
        depth=40,
        caption="1×1 Conv"
    ),

    # Sigmoid activation
    to_ConvSoftMax(
        name="out",
        s_filer="H×W",
        offset="(1.0,0,0)",
        to="(out_conv-east)",
        width=1,
        height=40,
        depth=40,
        caption="Sigmoid"
    ),

    # Connections
    to_connection("dec1", "out_conv"),
    to_connection("out_conv", "out"),

    # --- Horizontal visual legend ---

    to_ConvConvRelu(
        name='legend_conv',
        s_filer="",
        n_filer=("", ""),
        offset="(3.5,-7.0,0)",
        to="(0,0,0)",
        width=(1, 0),
        height=6,
        depth=6,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_conv-south) [below=8pt] {Conv + BatchNorm + ReLU};
    """,

    to_Pool(
        name="legend_pool",
        offset="(2.8,0,0)",
        to="(legend_conv-east)",
        width=1,
        height=6,
        depth=6,
        opacity=0.5,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_pool-south) [below=8pt] {MaxPool};
    """,

    to_UnPool(
        name="legend_up",
        offset="(2.4,0,0)",
        to="(legend_pool-east)",
        width=1,
        height=6,
        depth=6,
        opacity=0.5,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_up-south) [below=8pt] {Upsample};
    """,

    to_ConvSoftMax(
        name="legend_out",
        s_filer="",
        offset="(2.4,0,0)",
        to="(legend_up-east)",
        width=1,
        height=6,
        depth=6,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_out-south) [below=8pt] {Sigmoid};
    """,

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()