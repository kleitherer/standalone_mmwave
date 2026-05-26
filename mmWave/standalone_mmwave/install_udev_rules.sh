#!/usr/bin/env bash
# Install TI/MSP430 udev rules to /etc/udev/rules.d/ on the host.
# Run from standalone_mmwave: ./install_udev_rules.sh
# Requires sudo.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="$SCRIPT_DIR/config"
RULES_DEST="/etc/udev/rules.d"

if [[ ! -d "$RULES_SRC" ]]; then
  echo "Error: config/ not found at $RULES_SRC."
  exit 1
fi

for f in 61-msp430uif.rules 70-mm-no-ti-emulators.rules 71-ti-permissions.rules; do
  if [[ -f "$RULES_SRC/$f" ]]; then
    echo "Installing $f -> $RULES_DEST/"
    sudo cp "$RULES_SRC/$f" "$RULES_DEST/"
  else
    echo "Warning: $RULES_SRC/$f not found, skipping."
  fi
done

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Done. Verify with: ls -la /etc/udev/rules.d/ | grep -E '61-msp430|70-mm|71-ti'"
