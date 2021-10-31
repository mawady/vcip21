import json
import cv2
import numpy as np
import pywt
import uuid
import pickle
from shutil import copyfile
from os.path import isfile
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


def embed_watermark_dwt_channel(srcChl, wtrChl, gainFactor):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(srcChl, "haar", mode="reflect")
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, "haar", mode="reflect")
    # print("cV2 --> min: {}, max: {}".format(np.min(cV2), np.max(cV2)))
    # cH2 += wtrChl * gainFactor
    # cV2 += wtrChl * gainFactor
    # cH2 = cH2 * (1 + wtrChl * gainFactor)
    # cV2 = cV2 * (1 + wtrChl * gainFactor)
    # cD2 = cD2 * (1 + wtrChl * gainFactor)
    cA2 = cA2 * (1 + wtrChl * gainFactor)
    # print("cV2 --> min: {}, max: {}".format(np.min(cV2), np.max(cV2)))
    cA1 = pywt.idwt2((cA2, (cH2, cV2, cD2)), "haar", mode="reflect")
    embChl = pywt.idwt2((cA1, (cH1, cV1, cD1)), "haar", mode="reflect")
    return embChl


def extract_watermark_dwt_channel(srcChl, embChl, gainFactor):
    cA1, (cH1, cV1, cD1) = pywt.dwt2(srcChl, "haar", mode="reflect")
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, "haar", mode="reflect")
    yA1, (yH1, yV1, yD1) = pywt.dwt2(embChl, "haar", mode="reflect")
    yA2, (yH2, yV2, yD2) = pywt.dwt2(yA1, "haar", mode="reflect")
    # e_1 = (yH2 - cH2) / (gainFactor + 1e-7)
    # e_2 = (yV2 - cV2) / (gainFactor + 1e-7)
    # extChl = (e_1 + e_2) / 2
    # e_1 = (yH2 - cH2) / (cH2 * gainFactor + 1e-7)
    # e_2 = (yV2 - cV2) / (cV2 * gainFactor + 1e-7)
    # extChl = (e_1 + e_2) / 2
    # e_1 = (yH2 - cH2) / (cH2 * gainFactor + 1e-7)
    # e_2 = (yV2 - cV2) / (cV2 * gainFactor + 1e-7)
    # e_3 = (yD2 - cD2) / (cD2 * gainFactor + 1e-7)
    # extChl = (e_1 + e_2 + e_3) / 3
    extChl = (yA2 - cA2) / (cA2 * gainFactor + 1e-7)
    return extChl


def embed_wateramrk_dwt(gainFactor=0.01, **kwargs):
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
        bEmb = embed_watermark_dwt_channel(bSrc, bWtr, gainFactor)
        gEmb = embed_watermark_dwt_channel(gSrc, gWtr, gainFactor)
        rEmb = embed_watermark_dwt_channel(rSrc, rWtr, gainFactor)
        imgEmb = cv2.merge([bEmb, gEmb, rEmb])
    elif len(imgSrc.shape) == 2 and len(imgWtr.shape) == 2:
        imgEmb = embed_watermark_dwt_channel(imgSrc, imgWtr, gainFactor)
    else:
        assert (len(imgSrc.shape) == 3 and len(imgWtr.shape) != 3) or (
            len(imgSrc.shape) != 3 and len(imgWtr.shape) == 3
        ), "Source and Watermark images should be both in RGB or Gray-scale"

    pickle.dump(
        imgEmb,
        open(kwargs.get("outPath_imgEmb") + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL,
    )
    imgEmb = img_normalize(imgEmb)
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


def extract_wateramrk_dwt(gainFactor=0.01, **kwargs):
    # imgSrc = pickle.load(open(inSrcImgPath + ".pkl", "rb"))
    # imgEmb = pickle.load(open(kwargs.get("outPath_imgEmb") + ".pkl", "rb"))
    imgSrc = cv2.imread(kwargs.get("outPath_imgSrc"))
    imgSrc = img_denormalize(imgSrc)
    imgEmbPkl = pickle.load(open(kwargs.get("outPath_imgEmb") + ".pkl", "rb"))
    imgEmb = cv2.imread(kwargs.get("outPath_imgEmb"))
    imgEmb = img_denormalize(imgEmb, to_range=(np.min(imgEmbPkl), np.max(imgEmbPkl)))
    # imgEmb = img_denormalize(imgEmb)
    # print("imgEmb --> {}".format(getImgStat(imgEmb)))
    # print("imgEmbPkl --> {}".format(getImgStat(imgEmbPkl)))

    if len(imgSrc.shape) == 3 and len(imgEmb.shape) == 3:
        (bSrc, gSrc, rSrc) = cv2.split(imgSrc)
        (bEmb, gEmb, rEmb) = cv2.split(imgEmb)
        bExt = extract_watermark_dwt_channel(bSrc, bEmb, gainFactor)
        gExt = extract_watermark_dwt_channel(gSrc, gEmb, gainFactor)
        rExt = extract_watermark_dwt_channel(rSrc, rEmb, gainFactor)
        imgExt = cv2.merge([bExt, gExt, rExt])
    elif len(imgSrc.shape) == 2 and len(imgEmb.shape) == 2:
        imgExt = extract_watermark_dwt_channel(imgSrc, imgEmb, gainFactor)
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


def do_singleRun_dwt(inFld, inFname_imgSrc, inFname_imgWtr, method, choice, outFld="./tmp/", gainFactor=0.4):
    # 0.02, 0.4
    # gainFactor = 0.9  # set gain factor
    uid = uuid.uuid4().hex
    # inFld = "./test_images/"
    # inFname_imgOrg = "lena.png"
    # inFname_imgWtr = "peppers.png"
    # inFld = "./kodak_imgs/"
    dct_paths = generate_paths(
        uid, method, inFld, outFld, inFname_imgSrc, inFname_imgWtr
    )
    embed_wateramrk_dwt(gainFactor, **dct_paths)
    ###########################################
    ###########################################
    # print([e.value for e in encoding])
    do_compress(dct_paths.get("outPath_imgEmb"), choice)
    ###########################################
    ###########################################
    extract_wateramrk_dwt(gainFactor, **dct_paths)
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
    gainFactor = 0.4
    method = "ADD_DWT"
    choice = encoding.COMP_BMSHJ_1
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
        ) and False:
            status = "Exist!"
        else:
            dct_results.update(
                do_singleRun_dwt(
                    inFld, inFname_imgSrc, inFname_imgWtr, method, choice, gainFactor
                )
            )
            status = "Executed!"
        print("src: {}, wtr: {}: {} \n".format(inFname_imgSrc, inFname_imgWtr, status))
    dct_results.update(get_summary(dct_results))
    with open(json_name, "w") as f:
        json.dump(dct_results, f)
