## Providing information about GPG signing keys

In general see the [GitHub documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification) about generating and maintaining your GPG keys for signing commits and releases. There are also tons of how-tos and tutorials out there about GPG and its best practices (which might change over time).

Here are a few pointers how to deal with your GPG keys in the context of DAPHNE. <br/>Keys need to be:
1. entered in your GitHub profile ([see GitHub documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification)).
2. kept up to date (remove and add again in GitHub after prolonging the expiry date).
3. kept up to date in the ``KEYS.txt`` file in the DAPHNE repo root directory.
4. updated on third party key servers you might have used.

Keyserver recommendation: As of February 2024 we recommend keyserver.ubuntu.com, keys.openpgp.org and keys.mailvelope.com. The latter two are recommended as they perform email verification. 

* To export your key information to ``KEYS.txt`` use these two commands (replace ``<your-email>`` with the email address associated with your GPG key:

``` bash
gpg --keyid-format=0xshort --list-key --with-fingerprint <your-email> | tee -a KEYS.txt
gpg --armor --export <your-email> | tee -a KEYS.txt
```

* If you are updating your key information in ``KEYS.txt``, open that file with a text editor to remove the old key after appending the new one.


