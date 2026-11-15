# Deploying under mernstackdev.com/ai-roadmap/

Build with the three environment variables, then upload the **contents** of `_site/` to the `/ai-roadmap/` folder.

```bash
SITE_URL=https://mernstackdev.com/ai-roadmap \
PARENT_SITE_NAME=MernStackDev PARENT_SITE_URL=https://mernstackdev.com \
ADSENSE_CLIENT=ca-pub-6093608492043000 \
python scripts/build_site.py
```

`SITE_URL` must match the folder you upload to. Canonical links, the sitemap, `robots.txt` and `llms.txt` all use it, so a mismatch
tells search engines to index pages that do not exist.

Server-side steps only you can do (a static build cannot do them):

1. **Keep WordPress out of the folder.** If a roadmap URL ever redirects to a blog page (for example `/learn/` going to `/learn-mern/`),
   WordPress is catching it. The rule in `htaccess-redirect.txt` serves real files and folders first. Put it in the `.htaccess` in the
   web root, **above** the WordPress block.
2. **Root files.** `robots.txt` and `llms.txt` only count at the domain root (`mernstackdev.com/robots.txt`). Search engines ignore them in a
   sub-folder. Merge `robots-additions.txt` into the root `robots.txt` and add `llms-additions.txt` to the root `llms.txt`
   (it exists today and lists only blog posts).
3. **Search consoles.** Submit `https://mernstackdev.com/ai-roadmap/sitemap.xml` in Google Search Console and Bing Webmaster Tools.

If you ever move the site to a different folder, rebuild with the new `SITE_URL` and add a permanent redirect from the old folder to the new one.
