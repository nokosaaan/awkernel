#!/usr/bin/env bash
# Awkernel 実機 PXE ホスト（Legion）の恒久設定。1 回だけ実行する。
# 根拠: Notion 2026-11-19「DAG可視化統合とPXE運用簡略化」3 節
# 実行: bash scripts/pxe_host_setup.sh   （途中で sudo のパスワードを聞かれる）
# 実機の LAN 抜き差しのたびに dnsmasq の再起動や SSH の設定をやり直さずに済むようにする。
set -euo pipefail

CONF=/etc/dnsmasq.d/awkernel-pxe.conf
CON="有線接続 1"
TS=$(date +%Y%m%d-%H%M%S)

echo "== 1. dnsmasq: bind-interfaces -> bind-dynamic, dhcp-ignore=tag:!known"
sudo cp -a "$CONF" "$CONF.bak-$TS"
sudo sed -i 's/^bind-interfaces$/bind-dynamic/' "$CONF"
grep -q '^dhcp-ignore=tag:!known$' "$CONF" || echo 'dhcp-ignore=tag:!known' | sudo tee -a "$CONF" >/dev/null
sudo dnsmasq --test --conf-file="$CONF"
sudo systemctl enable dnsmasq
sudo systemctl restart dnsmasq
systemctl is-active dnsmasq

echo "== 2. NetworkManager: $CON を既定経路にしない（固定IP・自動接続は設定済み）"
sudo nmcli con modify "$CON" ipv4.never-default yes connection.autoconnect yes
nmcli -f ipv4.addresses,ipv4.never-default,connection.autoconnect con show "$CON"

echo "== 3. ~/.ssh/config: 実機用エントリを固定（既存のパスフレーズなし鍵を使用）"
cp -a ~/.ssh/config ~/.ssh/config.bak-$TS
python3 - <<'EOF'
import pathlib, re
p = pathlib.Path.home() / ".ssh/config"
text = p.read_text()
# 既存の 192.168.10.10 / 192.168.10.10] ブロックを除去
blocks = re.split(r'(?m)^(?=Host\s)', text)
keep = [b for b in blocks if not re.match(r'Host\s+(awkernel-target\s+)?192\.168\.10\.10\]?\s*$', b.splitlines()[0] if b.strip() else '')]
new = """Host awkernel-target 192.168.10.10
  HostName 192.168.10.10
  User awkernel
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  BatchMode yes
  StrictHostKeyChecking accept-new
  ConnectTimeout 5
"""
p.write_text(new + "\n" + "".join(keep).lstrip("\n"))
EOF
cat ~/.ssh/config

echo
echo "完了。バックアップ: $CONF.bak-$TS / ~/.ssh/config.bak-$TS"
echo "鍵が実機 Windows に未登録なら、実機側で次の公開鍵を"
echo "  C:\\ProgramData\\ssh\\administrators_authorized_keys に追記する:"
cat ~/.ssh/id_ed25519.pub
