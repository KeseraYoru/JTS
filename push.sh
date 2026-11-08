#!/usr/bin/env bash
# ============================================================
# push.sh —— 把代码/文档推到 https://github.com/KeseraYoru/JTS.git
#
# 用法：
#   bash push.sh                 # 体检 + 提交 + 推送
#   bash push.sh "提交信息"       # 自定义提交信息
#   CHECK_ONLY=1 bash push.sh    # 只体检，不提交不推送
#
# 为什么需要这个脚本：
#   仓库里曾经提交过 dl/convnext_base_22k_1k_224.pth（338M），
#   GitHub 单文件硬上限 100M，push 必被拒。历史已于 2026-11-08 重写剔除，
#   这个脚本负责「下次别再犯」——推送前拦住任何 >50M 的对象。
# ============================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export GIT_PAGER=cat

REMOTE="origin"
BRANCH="master"
LIMIT=$((50 * 1024 * 1024))          # 50M 告警线（GitHub 硬上限 100M）
MSG="${1:-chore: 同步训练脚本与文档}"
FAIL=0

echo "============================================================"
echo "① 体检：待提交文件里有没有大文件"
echo "============================================================"
STAGED_BIG=0
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
  if (( sz > LIMIT )); then
    printf '  ✗ %8.1f MB  %s\n' "$(echo "$sz/1048576" | bc -l)" "$f"
    STAGED_BIG=1; FAIL=1
  fi
done < <(git status --porcelain | awk '$1!="D"{print substr($0,4)}' | tr -d '"')

if (( STAGED_BIG == 0 )); then echo "  ✓ 无超限文件"; fi

echo
echo "============================================================"
echo "② 体检：整个 git 历史里的最大对象"
echo "============================================================"
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk -v lim="$LIMIT" '$1=="blob" && $3>lim {printf "  ✗ %.1f MB  %s\n", $3/1048576, $4; bad=1} END{if(!bad) print "  ✓ 历史里最大的对象也在 50M 以内"}'
HIST_BIG=$(git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk -v lim="$LIMIT" '$1=="blob" && $3>lim' | wc -l | tr -d ' ')
(( HIST_BIG > 0 )) && FAIL=1

echo
echo "============================================================"
echo "③ 体检：仓库总体积"
echo "============================================================"
printf "  .git 体积：%s（健康值 < 20M）\n" "$(du -sh .git | cut -f1)"
echo "  已跟踪文件数：$(git ls-files | wc -l | tr -d ' ')"

if (( FAIL == 1 )); then
  echo
  echo "!! 体检不通过，已中止（push 上去会被 GitHub 拒绝，或把仓库撑爆）。"
  echo "   处理办法：把该文件加进 .gitignore，然后"
  echo "     git rm --cached <文件> && git commit -m 'chore: 移出大文件'"
  echo "   如果它已经在历史里，需要重写历史（参见本次 2026-11-08 的做法）。"
  exit 1
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo
  echo "CHECK_ONLY=1，体检通过，未提交未推送。"
  exit 0
fi

echo
echo "============================================================"
echo "④ 提交"
echo "============================================================"
if [[ -z "$(git status --porcelain)" ]]; then
  echo "  工作区干净，无需提交"
else
  git add -A
  git commit -q -m "$MSG" && echo "  已提交：$(git log --oneline -1)"
fi

echo
echo "============================================================"
echo "⑤ 预检：能否连上 $REMOTE"
echo "============================================================"
if ! git ls-remote --heads "$REMOTE" >/dev/null 2>&1; then
  echo "  ✗ 连不上 ${REMOTE}（本机常见原因：GitHub 直连被墙，需要走代理）"
  echo
  echo "  两种解法，任选其一："
  echo "  A. 给 git 配代理（把 7890 换成你自己的端口，Clash 默认 7890，V2Ray 常见 1080）："
  echo "       git config --global http.proxy   http://127.0.0.1:7890"
  echo "       git config --global https.proxy  http://127.0.0.1:7890"
  echo "     配完重跑本脚本。用完想关掉："
  echo "       git config --global --unset http.proxy"
  echo "       git config --global --unset https.proxy"
  echo "  B. 改用 SSH（需要已上传公钥到 GitHub）："
  echo "       git remote set-url origin git@github.com:KeseraYoru/JTS.git"
  echo
  echo "  提交已经做好了，只是没推上去；网络恢复后直接 bash push.sh 即可。"
  exit 2
fi
echo "  ✓ 远端可达"

echo
echo "============================================================"
echo "⑥ 推送"
echo "============================================================"
if git push -u "$REMOTE" "$BRANCH"; then
  echo
  echo "  ✓ 推送成功：$(git remote get-url "$REMOTE") ($BRANCH)"
  echo "    本地 HEAD  $(git rev-parse --short HEAD)"
  echo "    远端 $BRANCH  $(git ls-remote "$REMOTE" "refs/heads/$BRANCH" 2>/dev/null | cut -c1-7)"
  echo
  echo "  注意：本仓库历史在 2026-11-08 重写过（剔除 338M 权重）。"
  echo "    若已 clone 过旧版本的设备，pull 会冲突 → 重新 clone 最省事；"
  echo "    若远端已有旧提交而被拒，才需要 git push -u --force ${REMOTE} ${BRANCH}。"
else
  echo "  ✗ 推送失败，常见两种："
  echo "    rejected（non-fast-forward）→ 历史重写过，需要 git push -u --force origin $BRANCH"
  echo "    RPC failed / timeout          → 网络问题，重跑本脚本"
  exit 3
fi
