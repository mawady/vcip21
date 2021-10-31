import os
import cv2
import numpy as np
from enum import Enum
import pickle
from tfci import compress, decompress
from functools import partial
from statistics import mean
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse


def get_metrics(imageA, imageB):
    val_ssim = ssim(imageA, imageB, multichannel=True)
    val_mse = mse(imageA, imageB)
    val_psnr = psnr(imageA, imageB)
    val_nrmse = nrmse(imageA, imageB)
    return {
        "ssim": round(val_ssim, 3),
        "mse": round(val_mse, 3),
        "psnr": round(val_psnr, 3),
        "nrmse": round(val_nrmse, 3),
    }


class normalize(Enum):
    OPENCV = 1
    NUMPY = 2


class soften(Enum):
    BLUR = 1
    MEDIAN = 2
    NONE = 3


class encoding(Enum):
    NO_COMP = "NO_COMP"
    COMP_TFCI_HI = "COMP_TFCI_HI"
    COMP_TFCI_MI = "COMP_TFCI_MI"
    COMP_TFCI_LO = "COMP_TFCI_LO"
    COMP_JPEG_90 = "COMP_JPEG_90"
    COMP_JPEG_70 = "COMP_JPEG_70"
    COMP_JPEG_50 = "COMP_JPEG_50"
    COMP_BMSHJ_8 = "COMP_BMSHJ_8"
    COMP_BMSHJ_4 = "COMP_BMSHJ_4"
    COMP_BMSHJ_1 = "COMP_BMSHJ_1"


def img_normalize(img, choice=normalize.OPENCV):
    if choice == normalize.OPENCV:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return np.interp(img, (np.min(img), np.max(img)), (0.0, 255.0)).astype(np.uint8)


def img_denormalize(img, choice=normalize.OPENCV, to_range=None):
    if to_range is None:
        rng_min = 0.0
        rng_max = 1.0
    else:
        rng_min = to_range[0]
        rng_max = to_range[1]
    if choice == normalize.OPENCV:
        return cv2.normalize(np.float32(img), None, rng_min, rng_max, cv2.NORM_MINMAX)
    return np.interp(np.float32(img), (np.min(img), np.max(img)), (rng_min, rng_max))


def img_soften(img, choice=soften.BLUR):
    if choice == soften.BLUR:  # DWT_HH
        return cv2.blur(np.float32(img), (3, 3))
    elif choice == soften.MEDIAN:  # DWT_HL, DWT_LH, DWT_LL
        return cv2.medianBlur(np.float32(img), 5)
    return img


def getImgStat(img):
    return {"mean": np.mean(img), "min": np.min(img), "max": np.max(img)}


def generate_paths(uid, method, inFld, outFld, inFname_imgSrc, inFname_imgWtr):
    dct_paths = dict()
    dct_paths["inPath_imgSrc"] = os.path.join(inFld, inFname_imgSrc)
    dct_paths["inPath_imgWtr"] = os.path.join(inFld, inFname_imgWtr)
    dct_paths["outPath_imgSrc"] = os.path.join(
        outFld, "_".join([uid, method, "_imgSrc_", inFname_imgSrc])
    )
    dct_paths["outPath_imgWtr"] = os.path.join(
        outFld, "_".join([uid, method, "_imgWtr_", inFname_imgWtr])
    )
    dct_paths["outPath_imgWtrFilt"] = os.path.join(
        outFld, "_".join([uid, method, "_imgWtrFilt_", inFname_imgWtr])
    )
    dct_paths["outPath_imgEmb"] = os.path.join(
        outFld, "_".join([uid, method, "_imgEmb_", inFname_imgSrc])
    )
    dct_paths["outPath_imgEmbComp"] = os.path.join(
        outFld, "_".join([uid, method, "_imgEmbComp_", inFname_imgSrc])
    )
    dct_paths["outPath_imgExt"] = os.path.join(
        outFld, "_".join([uid, method, "_imgExt_", inFname_imgWtr])
    )
    dct_paths["outPath_imgExtFilt"] = os.path.join(
        outFld, "_".join([uid, method, "_imgExtFilt_", inFname_imgWtr])
    )
    dct_paths["outPath_imgDiffSrc"] = os.path.join(
        outFld, "_".join([uid, method, "_imgDiffSrc_", inFname_imgSrc])
    )
    dct_paths["outPath_imgDiffWtr"] = os.path.join(
        outFld, "_".join([uid, method, "_imgDiffWtr_", inFname_imgWtr])
    )
    return dct_paths


def getDiffImgs(choice=soften.BLUR, **kwargs):
    imgSrc = pickle.load(open(kwargs.get("outPath_imgSrc") + ".pkl", "rb"))
    # imgEmb = pickle.load(open(kwargs.get("outPath_imgEmb") + ".pkl", "rb"))
    imgEmbPkl = pickle.load(open(kwargs.get("outPath_imgEmb") + ".pkl", "rb"))
    imgEmb = cv2.imread(kwargs.get("outPath_imgEmb"))
    imgEmb = img_denormalize(imgEmb, to_range=(np.min(imgEmbPkl), np.max(imgEmbPkl)))
    imgDiffSrc = np.abs(imgEmb - imgSrc)
    imgDiffSrc = img_normalize(imgDiffSrc)
    cv2.imwrite(kwargs.get("outPath_imgDiffSrc"), imgDiffSrc)
    imgSrc = img_normalize(imgSrc)
    imgEmb = img_normalize(imgEmb)
    dct_metrics_emb_src = get_metrics(imgEmb, imgSrc)
    print("Emb vs Src : {}".format(dct_metrics_emb_src))

    imgWtrRz = pickle.load(open(kwargs.get("outPath_imgWtr") + ".pkl", "rb"))
    imgExt = pickle.load(open(kwargs.get("outPath_imgExt") + ".pkl", "rb"))
    imgExtFilt = img_soften(imgExt, choice)
    imgWtrRzFilt = img_soften(imgWtrRz, choice)
    imgDiffwtr = np.abs(imgExtFilt - imgWtrRzFilt)
    imgDiffwtr = img_normalize(imgDiffwtr)
    cv2.imwrite(kwargs.get("outPath_imgDiffWtr"), imgDiffwtr)

    pickle.dump(
        imgWtrRzFilt,
        open(kwargs.get("outPath_imgWtrFilt") + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL,
    )
    # imgWtrRzFilt = cv2.normalize(imgWtrRzFilt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    imgWtrRzFilt = img_normalize(imgWtrRzFilt)
    cv2.imwrite(kwargs.get("outPath_imgWtrFilt"), imgWtrRzFilt)
    pickle.dump(
        imgExtFilt,
        open(kwargs.get("outPath_imgExtFilt") + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL,
    )
    # imgExtFilt = cv2.normalize(imgExtFilt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    imgExtFilt = img_normalize(imgExtFilt)
    cv2.imwrite(kwargs.get("outPath_imgExtFilt"), imgExtFilt)
    dct_metrics_ext_wtr = get_metrics(imgExtFilt, imgWtrRzFilt)
    print("Ext vs Wtr : {}".format(dct_metrics_ext_wtr))

    return dct_metrics_emb_src, dct_metrics_ext_wtr
    # return {
    #     "metrics_ext_wtr": dct_metrics_ext_wtr,
    #     "metrics_emb_src": dct_metrics_emb_src,
    # }


def compress_deepEncoding(imgPath, mode="hific-hi"):
    compress(mode, imgPath, imgPath + ".tfci")
    decompress(imgPath + ".tfci", imgPath)


def compress_JpegEnconding(imgPath, quality=90):
    imgemb = cv2.imread(imgPath)
    cv2.imwrite(
        imgPath + ".jpg",
        imgemb,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    imgemb = cv2.imread(imgPath + ".jpg")
    cv2.imwrite(imgPath, imgemb)


def compress_nothing():
    pass


def do_compress(imgPath, choice=encoding.NO_COMP):
    switcher = {
        encoding.COMP_JPEG_50: partial(compress_JpegEnconding, imgPath, quality=50),
        encoding.COMP_JPEG_70: partial(compress_JpegEnconding, imgPath, quality=70),
        encoding.COMP_JPEG_90: partial(compress_JpegEnconding, imgPath, quality=90),
        encoding.COMP_TFCI_HI: partial(compress_deepEncoding, imgPath, mode="hific-hi"),
        encoding.COMP_TFCI_MI: partial(compress_deepEncoding, imgPath, mode="hific-mi"),
        encoding.COMP_TFCI_LO: partial(compress_deepEncoding, imgPath, mode="hific-lo"),
        encoding.COMP_BMSHJ_8: partial(
            compress_deepEncoding, imgPath, mode="bmshj2018-factorized-mse-8"
        ),
        encoding.COMP_BMSHJ_4: partial(
            compress_deepEncoding, imgPath, mode="bmshj2018-factorized-mse-4"
        ),
        encoding.COMP_BMSHJ_1: partial(
            compress_deepEncoding, imgPath, mode="bmshj2018-factorized-mse-1"
        ),
        encoding.NO_COMP: partial(compress_nothing),
    }

    return switcher.get(choice, compress_nothing)()


def get_summary(dct_results):
    lst_ssim_src = list()
    lst_ssim_wtr = list()
    lst_psnr_src = list()
    lst_psnr_wtr = list()
    for key, value in dct_results.items():
        if "metrics_emb_src" in value:
            lst_ssim_src.append(value["metrics_emb_src"]["ssim"])
            lst_psnr_src.append(value["metrics_emb_src"]["psnr"])
            lst_ssim_wtr.append(value["metrics_ext_wtr"]["ssim"])
            lst_psnr_wtr.append(value["metrics_ext_wtr"]["psnr"])
    dct_summary = {
        "summary": {
            "ssim_emb_src": round(mean(lst_ssim_src), 3),
            "psnr_emb_src": round(mean(lst_psnr_src), 3),
            "ssim_ext_wtr": round(mean(lst_ssim_wtr), 3),
            "psnr_ext_wtr": round(mean(lst_psnr_wtr), 3),
        }
    }
    print(dct_summary)
    return dct_summary
