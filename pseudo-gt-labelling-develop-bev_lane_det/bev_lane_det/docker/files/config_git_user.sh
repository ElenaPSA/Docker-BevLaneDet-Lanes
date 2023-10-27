#!/usr/bin/env bash

# LDAP config
[ -z "${LDAP_URI}" ] &&               LDAP_URI='ldaps://ldap.probayes.net:636'
[ -z "${LDAP_USERSDN}" ] &&           LDAP_USERSDN='ou=users,dc=probayes,dc=com'
[ -z "${LDAP_USER_CLASS}" ] &&        LDAP_USER_CLASS='posixAccount'
[ -z "${LDAP_USERID_ATTR}" ] &&       LDAP_USERID_ATTR='uid'
[ -z "${LDAP_NAME_ATTR}" ] &&         LDAP_NAME_ATTR='cn'
[ -z "${LDAP_MAIL_ATTR}" ] &&      	  LDAP_MAIL_ATTR='mail'


# Check args
if [ $# -ne 1 -o "${1}" = '-h' -o "${1}" = '--help' ]; then
    echo "Usage: $0 <login>" 
    exit 0
fi
LDAP_UID="${1}" 

# Build LDAP parameters
LDAP_USER_FILTER="(&(objectClass=${LDAP_USER_CLASS})(${LDAP_USERID_ATTR}=${LDAP_UID}))" 

# Search for user & cache
full_user_name=$(ldapsearch -x -u -LLL \
	-o ldif-wrap=no \
    -H "${LDAP_URI}" \
    -b "${LDAP_USERSDN}" \
    "${LDAP_USER_FILTER}" \
    "${LDAP_NAME_ATTR}" \
    | grep "^\s*${LDAP_NAME_ATTR}:" \
    | sed "s/^\s*${LDAP_NAME_ATTR}:\s*//")

full_user_mail=$(ldapsearch -x -u -LLL \
	-o ldif-wrap=no \
    -H "${LDAP_URI}" \
    -b "${LDAP_USERSDN}" \
    "${LDAP_USER_FILTER}" \
    "${LDAP_MAIL_ATTR}" \
    | grep "^\s*${LDAP_MAIL_ATTR}:" \
    | sed "s/^\s*${LDAP_MAIL_ATTR}:\s*//")

echo $full_user_mail
echo $full_user_name

git config --global user.email $full_user_mail
git config --global user.name ${full_user_name}

git config --global diff.submodule log
git config --global fetch.recurseSubmodules on-demand
git config --global status.submoduleSummary true

exit 0
