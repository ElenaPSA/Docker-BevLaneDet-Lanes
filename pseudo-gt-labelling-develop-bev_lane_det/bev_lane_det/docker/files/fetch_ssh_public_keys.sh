#!/usr/bin/env bash

# Some constants
TOOL_NAME='ldap4ssh'

# Source system config if exists
[ -z "${SYSTEM_CONFIG_FILE}" ] && SYSTEM_CONFIG_FILE='/etc/default/ldap_authentication.conf'
[ -f "${SYSTEM_CONFIG_FILE}" ] && . "${SYSTEM_CONFIG_FILE}"

# LDAP config
[ -z "${ALLOWED_GROUPS}" ] &&         ALLOWED_GROUPS='it'
[ -z "${NO_SEARCH_USERS}" ] &&        NO_SEARCH_USERS='root,git'
[ -z "${LDAP_URI}" ] &&               LDAP_URI='ldaps://ldap.probayes.net:636'
[ -z "${LDAP_USERSDN}" ] &&           LDAP_USERSDN='ou=users,dc=probayes,dc=com'
[ -z "${LDAP_USER_CLASS}" ] &&        LDAP_USER_CLASS='posixAccount'
[ -z "${LDAP_USERID_ATTR}" ] &&       LDAP_USERID_ATTR='uid'
[ -z "${LDAP_USERSSH_ATTR}" ] &&      LDAP_USERSSH_ATTR='sshPublicKey'
[ -z "${LDAP_GROUPSDN}" ] &&          LDAP_GROUPSDN='ou=groups,dc=probayes,dc=com'
[ -z "${LDAP_GROUP_CLASS}" ] &&       LDAP_GROUP_CLASS='posixGroup'
[ -z "${LDAP_GROUPID_ATTR}" ] &&      LDAP_GROUPID_ATTR='cn'
[ -z "${LDAP_GROUPMEMBERS_ATTR}" ] && LDAP_GROUPMEMBERS_ATTR='memberUid'

# Some other config
[ -z "${PREVENT_LOCAL_AUTHKEYS}" ] && PREVENT_LOCAL_AUTHKEYS=0
[ -z "${LOCAL_AUTHKEYS_FILE}" ] &&    LOCAL_AUTHKEYS_FILE='.ssh/authorized_keys'
[ -z "${LDAP_HOME_DIR}" ] &&          LDAP_HOME_DIR='/home/ldap'
[ -z "${CACHE_DIR}" ] &&              CACHE_DIR="/var/cache/${TOOL_NAME}"


# Check args
if [ $# -ne 1 -o "${1}" = '-h' -o "${1}" = '--help' ]; then
    echo "Usage: $0 <login>" 
    exit 0
fi
LDAP_UID="${1}" 

if echo "${NO_SEARCH_USERS}" | grep -E "(^|,)${LDAP_UID}(,|$)" &>/dev/null; then
	exit 0
fi

# Build LDAP parameters
LDAP_USER_FILTER="(&(objectClass=${LDAP_USER_CLASS})(${LDAP_USERID_ATTR}=${LDAP_UID}))" 
if echo "${ALLOWED_GROUPS}" | grep ',' >/dev/null ; then
	ALLOWED_GROUPS_FILTER="|(${LDAP_GROUPID_ATTR}=$(echo "${ALLOWED_GROUPS}" | sed "s/,/)(${LDAP_GROUPID_ATTR}=/g"))"
else
	ALLOWED_GROUPS_FILTER="${LDAP_GROUPID_ATTR}=${ALLOWED_GROUPS}"
fi
LDAP_GROUP_FILTER="(&(objectClass=${LDAP_GROUP_CLASS})(${ALLOWED_GROUPS_FILTER}))"

# Check cache dir
[ -d "${CACHE_DIR}" ] || mkdir -p "${CACHE_DIR}"
CACHE_FILE="${CACHE_DIR}/${LDAP_UID}.pub"

# invalidate old cache
if [ -f "${CACHE_FILE}" ]; then
	find \
		"${CACHE_FILE}" \
		-maxdepth 1 \
		-type f \
		-mtime +7 \
		-delete
fi

# First search to check connection to LDAP server
ldapsearch -x -u -LLL \
    -H "${LDAP_URI}" \
    -b "${LDAP_USERSDN}" \
    "${LDAP_USER_FILTER}" \
    "${LDAP_USERSSH_ATTR}" \
	&>/dev/null

if [ $? -ne 0 ]; then	
	logger -s -t "${TOOL_NAME}" -p 'user.warning' \
		"Unable to connect to LDAP server. Trying cache."
	if [ -f "${CACHE_FILE}" ]; then
		logger -s -t "${TOOL_NAME}" \
			"Found cache for user ${LDAP_UID}."
		cat "${CACHE_FILE}"
	else
		logger -s -t "${TOOL_NAME}" -p 'user.warning' \
			"No cache found for user ${LDAP_UID}. Exiting."
		exit 1
	fi
	exit 0
fi

# Search for allowed groups containing requested user id
ldapsearch -x -u -LLL \
    -o ldif-wrap=no \
    -H "${LDAP_URI}" \
    -b "${LDAP_GROUPSDN}" \
    "${LDAP_GROUP_FILTER}" \
    "${LDAP_GROUPMEMBERS_ATTR}" \
    | grep "^\s*${LDAP_GROUPMEMBERS_ATTR}:" \
    | sed "s/^\s*${LDAP_GROUPMEMBERS_ATTR}:\s*//" \
    | grep "${LDAP_UID}" \
	&>/dev/null

if [ $? -ne 0 ]; then
	logger -s -t "${TOOL_NAME}" -p 'user.warning' \
		"No user ${LDAP_UID} in group(s) ${ALLOWED_GROUPS}. Exiting."
	exit 1
fi

# Check if user is locked using sambaAcctFlags
ldapsearch -x -u -LLL \
	-o ldif-wrap=no \
	-H "${LDAP_URI}" \
	-b "${LDAP_USERSDN}" \
	"${LDAP_USER_FILTER}" \
	'sambaAcctFlags' \
	| grep "^\s*sambaAcctFlags:" \
	| tr -d ' ' | cut -d':' -f2 | grep -F 'D'
if [ $? -eq 0 ]; then
	logger -s -t "${TOOL_NAME}" -p 'user.warning' \
		"The account of user ${LDAP_UID} seems to be disabled. Exiting."
	exit 1
fi

# Search for user & cache
ldapsearch -x -u -LLL \
	-o ldif-wrap=no \
    -H "${LDAP_URI}" \
    -b "${LDAP_USERSDN}" \
    "${LDAP_USER_FILTER}" \
    "${LDAP_USERSSH_ATTR}" \
    | grep "^\s*${LDAP_USERSSH_ATTR}:" \
    | sed "s/^\s*${LDAP_USERSSH_ATTR}:\s*//" \
	> "${CACHE_FILE}.new"

if [ $? -eq 0 ]; then
	mv -f "${CACHE_FILE}.new" "${CACHE_FILE}"
	logger -s -t "${TOOL_NAME}" \
		"Fetched SSH keys for user ${LDAP_UID}."
else
	logger -s -t "${TOOL_NAME}" -p 'user.warning' \
		"Unable to fetch SSH keys for user ${LDAP_UID}."
fi

# Show cache
if [ -f "${CACHE_FILE}" ]; then
	LOCAL_AUTHKEYS_PATH="${LDAP_HOME_DIR}/${LDAP_UID}/${LOCAL_AUTHKEYS_FILE}"
	if [ "${PREVENT_LOCAL_AUTHKEYS}" -eq 0 -a -r "${LOCAL_AUTHKEYS_PATH}" ]; then
		echo '### Local authorized_keys:'
		cat "${LOCAL_AUTHKEYS_PATH}"
	fi
	echo '### LDAP authorized_keys:'
	cat "${CACHE_FILE}"
else
	logger -s -t "${TOOL_NAME}" -p 'user.warning' \
		"Unable to cache SSH keys for user ${LDAP_UID}."
fi

exit 0
