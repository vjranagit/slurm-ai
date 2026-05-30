#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /etc/munge/munge.key ]; then
  /usr/sbin/create-munge-key
fi
chown munge:munge /etc/munge/munge.key
chmod 0400 /etc/munge/munge.key

mkdir -p /run/munge /run/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
mkdir -p /var/log/munge /var/lib/munge
chown -R munge:munge /etc/munge /run/munge /var/log/munge /var/lib/munge
chmod 0700 /etc/munge /run/munge /var/log/munge /var/lib/munge
chmod 0755 /run/munge
touch /var/log/munge/munged.log
chown munge:munge /var/log/munge/munged.log
chmod 0600 /var/log/munge/munged.log
chown -R slurm:slurm /run/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm

runuser -u munge -- /usr/sbin/munged --force
/usr/sbin/slurmctld -D &
SLURMCTLD_PID=$!
/usr/sbin/slurmd -D &

sleep 3
if grep -q "AccountingStorageType=accounting_storage/slurmdbd" /etc/slurm/slurm.conf; then
  sacctmgr -i add cluster local || true
  sacctmgr -i add account root Description="root" Organization="local" || true
  sacctmgr -i add user root Account=root || true
  sacctmgr -i add qos normal || true
  sacctmgr -i modify qos normal set MaxJobsPU=8 || true
fi

while true; do
  if ! kill -0 "$SLURMCTLD_PID" 2>/dev/null; then
    echo "slurmctld exited; stopping container"
    exit 1
  fi
  sleep 5
done
