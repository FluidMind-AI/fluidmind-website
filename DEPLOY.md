# Deploying fluidmind.ai

Static site (no build). Served by **Caddy** on the shared FluidMind droplet
`161.35.12.67`, alongside winepro.ai and vargaspartners.com — each isolated in
its own `/etc/caddy/sites-enabled/*.caddy`.

```sh
./scripts/deploy-droplet.sh        # rsync repo -> droplet
DRY=1 ./scripts/deploy-droplet.sh  # preview
```

- Web root on the box: `/var/www/fluidmind.ai` (owned `deploy:caddy`).
- Deploy identity: `~/.ssh/vargas_deploy_key` (the `deploy` user).
- Caddy config: `deploy/fluidmind.ai.caddy`, installed at
  `/etc/caddy/sites-enabled/fluidmind.ai.caddy`. Apex is canonical, `www`
  redirects to it; Caddy auto-provisions/renews the Let's Encrypt cert.
- Pre-DNS-cutover staging (real HTTPS): `https://fluidmind.161.35.12.67.nip.io/`.

The `CNAME` file is a GitHub Pages artifact and is not deployed to the droplet
(the deploy excludes it). It can be removed once GitHub Pages is retired.
