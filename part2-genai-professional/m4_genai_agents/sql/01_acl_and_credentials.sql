-- Demo 2, Step 4 (part 1): let the database user call OCI services, then create credentials.
-- Run as ADMIN in SQL Worksheet / Database Actions. Reference scripts based on the course demo:
-- verify names/parameters against the current Oracle Database 23ai docs before relying on them.

-- 1) Network ACL so the user can reach Object Storage, PAR links and the Generative AI endpoint
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace  => xs$ace_type(privilege_list => xs$name_list('connect'),
                        principal_name => 'ADMIN',
                        principal_type => xs_acl.ptype_db));
END;
/

-- 2) Credential for OCI services (values from Profile -> API keys config preview). Never commit real values.
BEGIN
  DBMS_CLOUD.CREATE_CREDENTIAL(
    credential_name => 'OCI_GENAI_CRED',
    user_ocid       => 'ocid1.user.oc1..replace_me',
    tenancy_ocid    => 'ocid1.tenancy.oc1..replace_me',
    private_key     => 'REPLACE_WITH_PRIVATE_KEY_BODY_WITHOUT_HEADER_AND_FOOTER',
    fingerprint     => 'aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99');
END;
/
