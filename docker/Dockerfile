FROM nvcr.io/nvidia/pytorch:22.11-py3

# install packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git vim libgtk2.0-dev tmux libmagickwand-dev ffmpeg libsm6 libxext6  && \
    rm -rf /var/cache/apk/*

# RUN conda install pytorch torchvision cudatoolkit=11.3 -c pytorch-nightly


##########################################################
#################SSH and LDAP Configuration###############
##########################################################
RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/log/supervisor
RUN mkdir /var/run/nscd
COPY files/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server libpam-ldap libnss-ldap ldap-utils nscd sudo cifs-utils vim nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy custom scripts
RUN mkdir -p /opt/bin/probayes
COPY files/fetch_ssh_public_keys.sh /opt/bin/probayes/fetch_ssh_public_keys.sh

RUN chmod a+x /opt/bin/probayes/fetch_ssh_public_keys.sh

# create dir for cache
RUN mkdir -p /var/cache/ldap4ssh

# copy and set config
COPY files/libnss-ldap.conf /etc/libnss-ldap.conf
COPY files/pam_ldap.conf /etc/pam_ldap.conf
COPY files/ldap.conf /etc/ldap.conf
COPY files/pam_mkhomedir.conf /usr/share/pam-configs/my_mkhomedir
RUN sed -i 's/^#AuthorizedKeysCommand .*$/AuthorizedKeysCommand \/opt\/bin\/probayes\/fetch_ssh_public_keys.sh/' /etc/ssh/sshd_config
RUN sed -i 's/^#AuthorizedKeysCommandUser .*$/AuthorizedKeysCommandUser root/' /etc/ssh/sshd_config
RUN sed -i 's/^#PasswordAuthentication .*$/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -E -i 's/^(passwd|group|shadow):.*$/& ldap/' /etc/nsswitch.conf
RUN mkdir -p /etc/ldap
RUN echo 'TLS_CACERT  /etc/ssl/certs/ca-certificates.crt' >> /etc/ldap/ldap.conf
RUN echo '%probayes   ALL=(ALL:ALL) ALL' >> /etc/sudoers.d/ldap
RUN pam-auth-update --package

# authorize group
RUN echo "ALLOWED_GROUPS='probayes'" >> /etc/default/ldap_authentication.conf

# start service ssh
RUN mkdir /var/run/sshd

COPY files/config_git_user.sh /opt/bin/probayes/config_git_user.sh
RUN echo 'bash /opt/bin/probayes/config_git_user.sh $USER' >> /etc/skel/.bashrc
RUN echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> /etc/skel/.bashrc
##########################################################
##########################################################
##########################################################

# Add Python path of main folder and submodule folder to correctly import packages
# that are in submodules. Necessary because submodule is a project directory, not a Python package
# RUN echo 'export PYTHONPATH="${PYTHONPATH}:/workspace/object_detection:/workspace/object_detection/lb_yolox"' >> /etc/skel/.bashrc

# Expose port for the ssh connection
EXPOSE 22
EXPOSE 8888

# Install Python dependencies
COPY requirements.txt /tmp
RUN pip --no-cache-dir install -r /tmp/requirements.txt
# RUN pre-commit install
# RUN pip install black flake8 mypy coverage bandit
# RUN wget -O /bin/hadolint https://github.com/hadolint/hadolint/releases/download/v1.16.3/hadolint-Linux-x86_64
# RUN chmod +x /bin/hadolint
# RUN ldconfig
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu

# Launch supervisord that starts services and keep the main process running
CMD ["/usr/bin/supervisord"]
