import json
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from enum import Enum
import uuid
import pickle
from shutil import copyfile
from image_utils import (
    img_normalize,
    img_denormalize,
    generate_paths,
    getDiffImgs,
    soften,
    img_soften,
    do_compress,
    encoding,
    get_summary,
)
from os.path import isfile

# from tfci import compress, decompress

# https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
# https://stackoverflow.com/questions/7110899/how-do-i-apply-a-dct-to-an-image-in-python
# https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct


class transform(Enum):
    OPENCV = 1
    SCIPY = 2


# implement 2D DCT
def dct2(a, choice=transform.OPENCV):
    if choice == transform.OPENCV:
        return cv2.dct(a)
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


# implement 2D IDCT
def idct2(a, choice=transform.OPENCV):
    if choice == transform.OPENCV:
        return cv2.idct(a)
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def embed_watermark_dct_channel(srcChl, wtrChl, gainFactor):
    H_Wtr = int(np.shape(srcChl)[0] / 4)
    W_Wtr = int(np.shape(srcChl)[1] / 4)
    i = 0
    H_from = i * H_Wtr
    H_to = (i + 1) * H_Wtr
    W_from = i * W_Wtr
    W_to = (i + 1) * W_Wtr
    srcChl_DCT = dct2(srcChl)

    srcChl_DCT[H_from:H_to, W_from:W_to] += (  # noqa: E203
        srcChl_DCT[H_from:H_to, W_from:W_to] * wtrChl * gainFactor  # noqa: E203
    )
    embChl = idct2(srcChl_DCT)
    return embChl


def extract_watermark_dct_channel(srcChl, embChl, gainFactor):
    H_Wtr = int(np.shape(srcChl)[0] / 4)
    W_Wtr = int(np.shape(srcChl)[1] / 4)
    i = 0
    H_from = i * H_Wtr
    H_to = (i + 1) * H_Wtr
    W_from = i * W_Wtr
    W_to = (i + 1) * W_Wtr

    srcChl_DCT = dct2(srcChl)
    embChl_DCT = dct2(embChl)

    extChl = (
        embChl_DCT[H_from:H_to, W_from:W_to]  # noqa: E203
        - srcChl_DCT[H_from:H_to, W_from:W_to]  # noqa: E203
    ) / (
        srcChl_DCT[H_from:H_to, W_from:W_to] * gainFactor + 1e-7  # noqa: E203
    )
    return extChl


def embed_wateramrk_dct(gainFactor=0.01, **kwargs):
    imgSrc = cv2.imread(kwargs.get("inPath_imgSrc"))
    imgWtr = cv2.imread(kwargs.get("inPath_imgWtr"))
    imgSrc = img_denormalize(imgSrc)
    imgWtr = img_denormalize(imgWtr)
    imgWtr = cv2.resize(
        imgWtr, (int(np.shape(imgSrc)[1] / 4), int(np.shape(imgSrc)[0] / 4))
    )
    if len(imgSrc.shape) == 3 and len(imgWtr.shape) == 3:
        (bSrc, gSrc, rSrc) = cv2.split(imgSrc)
        (bWtr, gWtr, rWtr) = cv2.split(imgWtr)
        bEmb = embed_watermark_dct_channel(bSrc, bWtr, gainFactor)
        gEmb = embed_watermark_dct_channel(gSrc, gWtr, gainFactor)
        rEmb = embed_watermark_dct_channel(rSrc, rWtr, gainFactor)
        imgEmb = cv2.merge([bEmb, gEmb, rEmb])
    elif len(imgSrc.shape) == 2 and len(imgWtr.shape) == 2:
        imgEmb = embed_watermark_dct_channel(imgSrc, imgWtr, gainFactor)
    else:
        assert (len(imgSrc.shape) == 3 and len(imgWtr.shape) != 3) or (
            len(imgSrc.shape) != 3 and len(imgWtr.shape) == 3
        ), "Source and Watermark images should be both in RGB or Gray-scale"

    # print("imgSrc --> {}".format(getImgStat(imgSrc)))
    # print("imgEmb --> {}".format(getImgStat(imgEmb)))

    pickle.dump(
        imgEmb,
        open(kwargs.get("outPath_imgEmb") + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL,
    )
    imgEmb = img_normalize(imgEmb)
    # print("imgEmb --> {}".format(getImgStat(imgEmb)))
    cv2.imwrite(kwargs.get("outPath_imgEmb"), imgEmb)

    if kwargs.get("outPath_imgSrc") is not None:
        pickle.dump(
            imgSrc,
            open(kwargs.get("outPath_imgSrc") + ".pkl", "wb"),
            pickle.HIGHEST_PROTOCOL,
        )
        # imgSrc = img_normalize(imgSrc)
        # cv2.imwrite(kwargs.get("outPath_imgSrc"), imgSrc)
        copyfile(kwargs.get("inPath_imgSrc"), kwargs.get("outPath_imgSrc"))

    if kwargs.get("outPath_imgWtr") is not None:
        pickle.dump(
            imgWtr,
            open(kwargs.get("outPath_imgWtr") + ".pkl", "wb"),
            pickle.HIGHEST_PROTOCOL,
        )
        imgWtr = img_normalize(imgWtr)
        cv2.imwrite(kwargs.get("outPath_imgWtr"), imgWtr)


def extract_wateramrk_dct(gainFactor=0.01, **kwargs):
    # imgSrc = pickle.load(open(inSrcImgPath + ".pkl", "rb"))
    # imgEmb = pickle.load(open(inEmbImgPath + ".pkl", "rb"))
    imgEmbPkl = pickle.load(open(kwargs.get("outPath_imgEmb") + ".pkl", "rb"))
    imgSrc = cv2.imread(kwargs.get("outPath_imgSrc"))
    imgSrc = img_denormalize(imgSrc)
    imgEmb = cv2.imread(kwargs.get("outPath_imgEmb"))
    # print("imgEmb --> {}".format(getImgStat(imgEmb)))
    imgEmb = img_denormalize(imgEmb, to_range=(np.min(imgEmbPkl), np.max(imgEmbPkl)))
    # imgEmb = img_denormalize(imgEmb)
    # print("imgEmb --> {}".format(getImgStat(imgEmb)))

    if len(imgSrc.shape) == 3 and len(imgEmb.shape) == 3:
        (bSrc, gSrc, rSrc) = cv2.split(imgSrc)
        (bEmb, gEmb, rEmb) = cv2.split(imgEmb)
        bExt = extract_watermark_dct_channel(bSrc, bEmb, gainFactor)
        gExt = extract_watermark_dct_channel(gSrc, gEmb, gainFactor)
        rExt = extract_watermark_dct_channel(rSrc, rEmb, gainFactor)
        imgExt = cv2.merge([bExt, gExt, rExt])
    elif len(imgSrc.shape) == 2 and len(imgEmb.shape) == 2:
        imgExt = extract_watermark_dct_channel(imgSrc, imgEmb, gainFactor)
    else:
        assert (len(imgSrc.shape) == 3 and len(imgEmb.shape) != 3) or (
            len(imgSrc.shape) != 3 and len(imgEmb.shape) == 3
        ), "Source and Embedded images should be both in RGB or Gray-scale"

    pickle.dump(
        imgExt,
        open(kwargs.get("outPath_imgExt") + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL,
    )
    imgExt = img_soften(imgExt, choice=soften.MEDIAN)
    imgExt = img_normalize(imgExt)
    cv2.imwrite(kwargs.get("outPath_imgExt"), imgExt)


def do_singleRun_dct(
    inFld,
    inFname_imgSrc,
    inFname_imgWtr,
    method,
    choice,
    outFld="./tmp/",
    gainFactor=0.9,
):
    # 0.02, 0.4
    # gainFactor = 0.9  # set gain factor
    uid = uuid.uuid4().hex
    # inFld = "./test_images/"
    # inFname_imgOrg = "lena.png"
    # inFname_imgWtr = "peppers.png"
    # inFld = "./kodak_imgs/"
    # outFld = "./tmp/"
    dct_paths = generate_paths(
        uid, method, inFld, outFld, inFname_imgSrc, inFname_imgWtr
    )
    embed_wateramrk_dct(gainFactor, **dct_paths)
    ###########################################
    ###########################################
    # print([e.value for e in encoding])
    do_compress(dct_paths.get("outPath_imgEmb"), choice)
    ###########################################
    ###########################################
    extract_wateramrk_dct(gainFactor, **dct_paths)
    dct_metrics_emb_src, dct_metrics_ext_wtr = getDiffImgs(
        choice=soften.MEDIAN, **dct_paths
    )
    return {
        "_".join([inFname_imgSrc, inFname_imgWtr, method, choice.value]): {
            "ID": uid,
            "srcImg": inFname_imgSrc,
            "wtrImg": inFname_imgWtr,
            "method": method,
            "compression": choice.value,
            "metrics_emb_src": dct_metrics_emb_src,
            "metrics_ext_wtr": dct_metrics_ext_wtr,
        }
    }


if __name__ == "__main__":
    gainFactor = 0.9
    method = "ADD_DCT"
    choice = encoding.NO_COMP
    inFld = "./kodak_imgs/"
    lst_srcImgs = [
        "kodim23.png",
        "kodim01.png",
        "kodim05.png",
        "kodim21.png",
        "kodim02.png",
    ]
    inFname_imgWtr = "kodim15.png"
    # inFname_imgSrc = "kodim02.png"
    json_name = "json/" + "_".join([method, choice.value]) + ".json"
    if isfile(json_name):
        with open(json_name, "r") as f:
            dct_results = json.load(f)
    else:
        dct_results = dict()
    for inFname_imgSrc in lst_srcImgs:
        if (
            "_".join([inFname_imgSrc, inFname_imgWtr, method, choice.value])
            in dct_results
        ):
            status = "Exist!"
        else:
            dct_results.update(
                do_singleRun_dct(
                    inFld, inFname_imgSrc, inFname_imgWtr, method, choice, gainFactor
                )
            )
            status = "Executed!"
        print("src: {}, wtr: {}: {} \n".format(inFname_imgSrc, inFname_imgWtr, status))
    dct_results.update(get_summary(dct_results))
    with open(json_name, "w") as f:
        json.dump(dct_results, f)
