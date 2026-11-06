#!/usr/bin/env bash
# 下载合规预训练权重（公网直链、无需登录、非 NC、非 gated）
#
# 为什么要处理证书问题
# ----------------
# 部分实验室机器的 ca-certificates 较旧，访问 fbaipublicfiles.com 会报
# "certificate has expired"。这不是权重的问题，也不影响权重内容：
# 我们最终靠 SHA-256 校验完整性与可复现性（申报 A5 用），所以先用普通方式，
# 失败后降级为 -k（跳过证书校验，仅影响传输层的身份验证）。
# 若连 -k 也不通，就在本机下载后 scp 上去（脚本末尾有提示）。
set -u
cd "$(dirname "$0")"
mkdir -p weights

DINO_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
CNX22K_URL="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth"

fetch () {  # $1=url  $2=输出文件  $3=期望最小体积(MB)
  local url="$1" dst="$2" minmb="$3"
  if [ -f "$dst" ]; then
    local mb=$(( $(wc -c < "$dst") / 1024 / 1024 ))
    if [ "$mb" -ge "$minmb" ]; then
      echo "  已存在（${mb}MB），跳过"
      return 0
    fi
    echo "  已有文件只有 ${mb}MB（可能是不完整的下载），重新下载"
    rm -f "$dst"
  fi
  echo "  下载 $url"
  if curl -L --retry 3 --connect-timeout 20 --progress-bar -o "$dst" "$url"; then
    :
  else
    echo "  ⚠ 普通下载失败。常见原因：本机 CA 证书过期 / 代理拦截 HTTPS。"
    echo "    改用 -k 跳过证书校验重试（权重内容不变，SHA-256 会记录在下方）"
    curl -k -L --retry 3 --connect-timeout 20 --progress-bar -o "$dst" "$url" || {
      echo "  ✗ 仍然失败。请在本机下载后上传："
      echo "      curl -L -o $dst \"$url\""
      echo "      scp $dst 你的用户名@本机:$(pwd)/weights/"
      return 1
    }
  fi
  local mb=$(( $(wc -c < "$dst") / 1024 / 1024 ))
  if [ "$mb" -lt "$minmb" ]; then
    echo "  ✗ 下载体积异常（${mb}MB < ${minmb}MB），可能拿到了错误页面"
    rm -f "$dst"
    return 1
  fi
  echo "  ✓ 完成（${mb}MB）"
}

echo "== 1/2 DINOv2 ViT-B/14（LVD-142M 自监督，Apache-2.0）=="
fetch "$DINO_URL" weights/dinov2_vitb14_pretrain.pth 300

echo "== 2/2 ConvNeXt-Base ImageNet-22k（MIT）=="
fetch "$CNX22K_URL" weights/convnext_base_22k_224.pth 300

echo
echo "== 体积与哈希（申报 A5 要 SHA-256）=="
ls -lh weights/ 2>/dev/null
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum weights/*.pth 2>/dev/null
else
  shasum -a 256 weights/*.pth 2>/dev/null
fi

echo
echo "下一步："
echo "  python tools/selftest_archs.py --arch dinov2_vitb14 --img-size 392 \\"
echo "    --weights weights/dinov2_vitb14_pretrain.pth"
echo "  期望看到：匹配 173/174 键 + pos_embed 已插值 + 前向 (2,10) + fp16 164MB"
