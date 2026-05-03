import sys
sys.path.append('../')

from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
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

    # --- FullyConv blocks (with activation strip) ---

    to_ConvConvRelu("gconv1","H×W",(32,""),"(1.5,0,0)","(events-east)",(3,0),40,40,"rate 1"),
    to_ConvConvRelu("gconv2","H×W",(32,""),"(1.3,0,0)","(gconv1-east)",(3,0),40,40,"rate 2"),
    to_ConvConvRelu("gconv3","H×W",(32,""),"(1.3,0,0)","(gconv2-east)",(3,0),40,40,"rate 4"),
    to_ConvConvRelu("gconv4","H×W",(32,""),"(1.3,0,0)","(gconv3-east)",(3,0),40,40,"rate 8"),
    to_ConvConvRelu("gconv5","H×W",(32,""),"(1.3,0,0)","(gconv4-east)",(3,0),40,40,"rate 16"),
    to_ConvConvRelu("gconv6","H×W",(32,""),"(1.3,0,0)","(gconv5-east)",(3,0),40,40,"rate 32"),
    to_ConvConvRelu("gconv7","H×W",(32,""),"(1.3,0,0)","(gconv6-east)",(3,0),40,40,"rate 64"),
    to_ConvConvRelu("gconv8","H×W",(32,""),"(1.3,0,0)","(gconv7-east)",(3,0),40,40,"rate 128"),
    to_ConvConvRelu("gconv9","H×W",(32,""),"(1.3,0,0)","(gconv8-east)",(3,0),40,40,"rate 1"),

    # Final 1x1 conv (no activation strip)
    to_Conv(
        name="out",
        s_filer="H×W",
        n_filer=3,
        offset="(1.5,0,0)",
        to="(gconv9-east)",
        width=3,
        height=40,
        depth=40,
        caption="1×1 Conv"
    ),

    # Connections
    to_connection("events", "gconv1"),
    to_connection("gconv1", "gconv2"),
    to_connection("gconv2", "gconv3"),
    to_connection("gconv3", "gconv4"),
    to_connection("gconv4", "gconv5"),
    to_connection("gconv5", "gconv6"),
    to_connection("gconv6", "gconv7"),
    to_connection("gconv7", "gconv8"),
    to_connection("gconv8", "gconv9"),
    to_connection("gconv9", "out"),

    # --- Legend ---

    to_ConvConvRelu(
        name='legend_conv',
        s_filer="",
        n_filer=("", ""),
        offset="(6.0,-8.0,0)",
        to="(0,0,0)",
        width=(1, 0.3),
        height=6,
        depth=6,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_conv-south) [below=8pt] {Dilated Conv + nm + LeakyReLU};
    """,

    to_Conv(
        name="legend_out",
        s_filer="",
        n_filer="",
        offset="(4.0,0,0)",
        to="(legend_conv-east)",
        width=1,
        height=6,
        depth=6,
        caption=""
    ),
    r"""
    \node[font=\small, align=center] 
    at (legend_out-south) [below=8pt] {1×1 Conv};
    """,

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()