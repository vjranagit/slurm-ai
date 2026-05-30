FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    munge \
    slurm-wlm \
    slurm-wlm-basic-plugins \
    slurm-client \
    tini \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
    /etc/slurm \
    /etc/munge \
    /run/munge \
    /var/lib/munge \
    /var/log/munge \
    /run/slurm \
    /var/spool/slurmd \
    /var/spool/slurmctld \
    /var/log/slurm \
    && chown -R munge:munge /etc/munge /run/munge /var/lib/munge /var/log/munge \
    && chmod 0700 /etc/munge /var/lib/munge /var/log/munge \
    && chmod 0755 /run/munge \
    && chown -R slurm:slurm /run/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm

COPY infra/etc/slurm/slurm.conf /etc/slurm/slurm.conf
COPY infra/etc/slurm/cgroup.conf /etc/slurm/cgroup.conf
COPY infra/docker/slurm/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6817 6818
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
