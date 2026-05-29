#!/bin/bash
# ============================================================================
#  setup_autostart.sh — Cài Baby AI Alert tự chạy khi bật Orange Pi
# ============================================================================
#  Chạy 1 LẦN DUY NHẤT. Sau đó mỗi lần cắm điện/boot, app tự chạy — KHÔNG
#  cần gõ thêm lệnh nào nữa.
#
#  Cách dùng (trên Orange Pi, trong thư mục project):
#       sudo bash scripts/setup_autostart.sh
#
#  Script tự động:
#    1. Phát hiện project dir + venv python + user/group cho đúng máy
#    2. Sinh /etc/systemd/system/baby-monitor.service từ template deploy/
#    3. Tạo file log + logrotate (giữ 7 ngày, tránh đầy đĩa)
#    4. Tạo cron dọn events/ cũ > 30 ngày
#    5. enable + start service → tự chạy từ giờ trở đi
#
#  Gỡ tự-chạy:  sudo bash scripts/setup_autostart.sh --uninstall
#  Idempotent — chạy lại nhiều lần OK.
# ============================================================================

set -e

SERVICE_NAME="baby-monitor"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_PATH="/var/log/baby-monitor.log"
LOGROTATE_PATH="/etc/logrotate.d/baby-monitor"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$PROJ_ROOT/deploy/baby-monitor.service"

# --- Phải chạy bằng sudo (cần ghi vào /etc/systemd) ---
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Cần quyền root. Chạy:  sudo bash scripts/setup_autostart.sh"
    exit 1
fi

# ============================================================================
#  Gỡ cài đặt
# ============================================================================
if [ "$1" = "--uninstall" ]; then
    echo "=== Gỡ autostart Baby AI Alert ==="
    systemctl stop  "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_PATH" "$LOGROTATE_PATH"
    systemctl daemon-reload
    echo "✅ Đã gỡ. App sẽ KHÔNG còn tự chạy khi boot."
    echo "   (file log $LOG_PATH và cron dọn events/ giữ nguyên — xoá tay nếu muốn)"
    exit 0
fi

echo "=========================================="
echo "Baby AI Alert — Cài autostart (systemd)"
echo "Project dir: $PROJ_ROOT"
echo "=========================================="

# --- Sanity: template tồn tại ---
if [ ! -f "$TEMPLATE" ]; then
    echo "❌ Không thấy template: $TEMPLATE"
    echo "   Bạn có đang ở đúng thư mục project (có thư mục deploy/) không?"
    exit 1
fi

# --- Phát hiện USER chạy app: ưu tiên người gọi sudo, KHÔNG chạy dưới root ---
RUN_USER="${SUDO_USER:-}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = "root" ]; then
    # Fallback: owner của thư mục project
    RUN_USER="$(stat -c '%U' "$PROJ_ROOT")"
fi
if [ "$RUN_USER" = "root" ]; then
    echo "⚠️  App sẽ chạy dưới root — không khuyến nghị. Nên tạo user thường"
    echo "   (xem INSTALL.md §2.3) rồi chạy lại script bằng user đó qua sudo."
fi
RUN_GROUP="video"   # cần group video để mở camera (INSTALL.md §4.2)

# --- Phát hiện python: ưu tiên venv của project ---
if [ -x "$PROJ_ROOT/venv/bin/python" ]; then
    PYTHON_BIN="$PROJ_ROOT/venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || true)"
    echo "⚠️  Không thấy venv ($PROJ_ROOT/venv). Dùng python hệ thống: $PYTHON_BIN"
    echo "   Khuyến nghị tạo venv trước (INSTALL.md §5)."
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Không tìm thấy python. Cài Python trước (INSTALL.md §5)."
    exit 1
fi

echo ""
echo "Cấu hình phát hiện được:"
echo "   User   : $RUN_USER"
echo "   Group  : $RUN_GROUP"
echo "   Python : $PYTHON_BIN"
echo "   Entry  : $PROJ_ROOT/src/main.py"
echo ""

# --- Verify user thuộc group video (nếu không, alert camera sẽ fail) ---
if id -nG "$RUN_USER" 2>/dev/null | grep -qw "video"; then
    echo "✅ User '$RUN_USER' đã thuộc group video."
else
    echo "⚠️  User '$RUN_USER' CHƯA thuộc group video → có thể không mở được camera."
    echo "   Đang thêm: usermod -aG video $RUN_USER"
    usermod -aG video "$RUN_USER" || true
    echo "   (cần logout/login lại user đó để áp dụng đầy đủ)"
fi

# ============================================================================
#  1. Sinh service file từ template
# ============================================================================
echo ""
echo "=== Tạo $SERVICE_PATH ==="
# Dùng | làm delimiter cho sed vì path chứa /
sed -e "s|__USER__|$RUN_USER|g" \
    -e "s|__GROUP__|$RUN_GROUP|g" \
    -e "s|__PROJECT_DIR__|$PROJ_ROOT|g" \
    -e "s|__PYTHON__|$PYTHON_BIN|g" \
    "$TEMPLATE" > "$SERVICE_PATH"
echo "✅ Đã ghi service file."

# ============================================================================
#  2. File log + quyền
# ============================================================================
echo ""
echo "=== Tạo file log $LOG_PATH ==="
touch "$LOG_PATH"
chown "$RUN_USER":"$RUN_GROUP" "$LOG_PATH"
echo "✅ Log: $LOG_PATH"

# ============================================================================
#  3. Logrotate — giữ 7 ngày, tránh đầy đĩa khi chạy 24/7
# ============================================================================
echo ""
echo "=== Cấu hình logrotate ==="
cat > "$LOGROTATE_PATH" << EOF
$LOG_PATH {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 $RUN_USER $RUN_GROUP
    postrotate
        systemctl restart $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
echo "✅ Logrotate: $LOGROTATE_PATH (7 ngày)"

# ============================================================================
#  4. Cron dọn events/ cũ > 30 ngày (tránh đầy đĩa snapshot)
# ============================================================================
echo ""
echo "=== Cron dọn events/ cũ ==="
CRON_LINE="0 2 * * * find $PROJ_ROOT/events -mtime +30 -delete"
CRON_TMP="$(mktemp)"
# Lấy crontab hiện tại của user (nếu có), bỏ dòng cũ trỏ tới project này, thêm lại
crontab -u "$RUN_USER" -l 2>/dev/null | grep -vF "$PROJ_ROOT/events" > "$CRON_TMP" || true
echo "$CRON_LINE" >> "$CRON_TMP"
crontab -u "$RUN_USER" "$CRON_TMP"
rm -f "$CRON_TMP"
echo "✅ Cron: dọn events/ > 30 ngày, mỗi 02:00 sáng (user $RUN_USER)"

# ============================================================================
#  5. enable + start
# ============================================================================
echo ""
echo "=== Kích hoạt service ==="
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "=========================================="
echo "✅ XONG! App đã tự chạy và sẽ TỰ CHẠY MỖI LẦN BOOT."
echo "   (không cần gõ lệnh gì nữa, kể cả sau khi mất điện/reboot)"
echo ""
echo "Kiểm tra trạng thái:"
echo "   sudo systemctl status $SERVICE_NAME"
echo "   journalctl -u $SERVICE_NAME -f        # xem log live"
echo "   tail -f $LOG_PATH"
echo ""
echo "Tạm dừng để dev (giải phóng camera):"
echo "   sudo systemctl stop $SERVICE_NAME"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "Gỡ tự-chạy:"
echo "   sudo bash scripts/setup_autostart.sh --uninstall"
echo "=========================================="

# In trạng thái ngay để người cài thấy kết quả
sleep 2
systemctl status "$SERVICE_NAME" --no-pager -l || true
