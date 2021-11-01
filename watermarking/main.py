import argparse
import os
from watermark_add_dct import do_singleRun_dct
from watermark_add_dwt import do_singleRun_dwt
from image_utils import encoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main")

    # general options
    parser.add_argument(
        "--inFolder",
        type=str,
        default="./kodak_imgs/",
        help="Input Folder (default: ./kodak_imgs/).",
    )
    parser.add_argument(
        "--outFolder",
        type=str,
        default="./tmp/",
        help="Output Folder (default: ./tmp/).",
    )
    parser.add_argument(
        "--imgSrc",
        type=str,
        required=True,
        help="File name of input image (i.e. kodim23.png).",
    )
    parser.add_argument(
        "--imgWtr",
        type=str,
        required=True,
        help="File name of watermark image (i.e. kodim23.png).",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Encoding/decoding method [ADD_DCT/ADD_DWT].",
    )
    parser.add_argument(
        "--comp",
        type=encoding,
        choices=list(encoding),
        required=True,
    )

    args = parser.parse_args()
    print("args:\n{}".format(args))

    if not os.path.exists(args.outFolder):
        os.makedirs(args.outFolder)

    if args.method == "ADD_DCT":
        results = do_singleRun_dct(
            args.inFolder,
            args.imgSrc,
            args.imgWtr,
            args.method,
            args.comp,
            args.outFolder
        )
    else:
        results = do_singleRun_dwt(
            args.inFolder,
            args.imgSrc,
            args.imgWtr,
            args.method,
            args.comp,
            args.outFolder
        )

    print("results:\n{}".format(results))
