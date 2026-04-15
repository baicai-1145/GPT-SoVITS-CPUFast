import traceback
from collections import OrderedDict
from time import time as ttime
import shutil
import os
import torch
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


from io import BytesIO

model_version2byte = {
    "v2Pro": b"05",
    "v2ProPlus": b"06",
}


def my_save2(fea, path, model_version):
    bio = BytesIO()
    torch.save(fea, bio)
    bio.seek(0)
    data = bio.getvalue()
    byte = model_version2byte[model_version]
    data = byte + data[2:]
    with open(path, "wb") as f:
        f.write(data)


def savee(ckpt, name, epoch, steps, hps, model_version=None, lora_rank=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = hps
        opt["info"] = "%sepoch_%siteration" % (epoch, steps)
        if lora_rank:
            opt["lora_rank"] = lora_rank
            my_save2(opt, "%s/%s.pth" % (hps.save_weight_dir, name), model_version)
        elif model_version != None and "Pro" in model_version:
            my_save2(opt, "%s/%s.pth" % (hps.save_weight_dir, name), model_version)
        else:
            my_save(opt, "%s/%s.pth" % (hps.save_weight_dir, name))
        return "Success."
    except:
        return traceback.format_exc()


"""
00:v1
01:v2
05:v2Pro
06:v2ProPlus
"""
head2version = {
    b"00": ["v1", "v1", False],
    b"01": ["v2", "v2", False],
    b"05": ["v2", "v2Pro", False],
    b"06": ["v2", "v2ProPlus", False],
}
hash_pretrained_dict = {
    "dc3c97e17592963677a4a1681f30c653": ["v2", "v2", False],  # s2G488k.pth#sovits_v1_pretrained
    "6642b37f3dbb1f76882b69937c95a5f3": ["v2", "v2", False],  # s2G2333K.pth#sovits_v2_pretrained
    "c7e9fce2223f3db685cdfa1e6368728a": ["v2", "v2Pro", False],  # s2Gv2Pro.pth#sovits_v2Pro_pretrained
    "66b313e39455b57ab1b0bc0b239c9d0a": ["v2", "v2ProPlus", False],  # s2Gv2ProPlus.pth#sovits_v2ProPlus_pretrained
}
import hashlib


def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def get_sovits_version_from_path_fast(sovits_path):
    ###1-if it is pretrained sovits models, by hash
    hash = get_hash_from_file(sovits_path)
    if hash in hash_pretrained_dict:
        return hash_pretrained_dict[hash]
    ###2-new weights, by head
    with open(sovits_path, "rb") as f:
        version = f.read(2)
    if version != b"PK":
        if version in {b"02", b"03", b"04"}:
            raise ValueError(i18n("当前仓库已移除 SoVITS v3/v4 支持"))
        return head2version[version]
    ###3-old weights, by file size
    size = os.path.getsize(sovits_path)
    """
            v1weights:about 82942KB
                half thr:82978KB
            v2weights:about 83014KB
            old unsupported large v3/v4 weights:about 750MB
    """
    if size < 82978 * 1024:
        model_version = version = "v1"
    elif size < 700 * 1024 * 1024:
        model_version = version = "v2"
    else:
        raise ValueError(i18n("当前仓库已移除 SoVITS v3/v4 支持"))
    return version, model_version, False


def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)
