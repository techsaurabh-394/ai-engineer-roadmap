# Deploying under mernstackdev.com/ai-engineer-roadmap/

Build with the three environment variables, then upload the **contents** of `_site/` to the `/ai-engineer-roadmap/` folder.

```bash
SITE_URL=https://mernstackdev.com/ai-engineer-roadmap \
PARENT_SITE_NAME=MernStackDev PARENT_SITE_URL=https://mernstackdev.com \
ADSENSE_CLIENT=ca-pub-6093608492043000 \
python scripts/build_site.py
```

Three server-side steps only you can do (a static build cannot do them):

1. **Keep the old URLs ranking.** The previous version lives at `/ai-roadmap/`. Every page there has the same path under
   `/ai-engineer-roadmap/`, so one permanent redirect keeps its links and rankings. Add the rule from `htaccess-redirect.txt`
   to the `.htaccess` in the web root, **above** the WordPress block.
2. **Do not let WordPress swallow the folder.** If `/ai-engineer-roadmap/learn/` redirects to a blog URL, WordPress is catching it. The
   rule in `htaccess-redirect.txt` serves real files and folders first, which fixes this.
3. **Root files.** `robots.txt` and `llms.txt` only count at the domain root (`mernstackdev.com/robots.txt`). A search engine ignores them in a
   sub-folder. Merge `robots-additions.txt` into the root `robots.txt` and add `llms-additions.txt` to the root `llms.txt`
   (it exists today and lists only blog posts). Then submit `https://mernstackdev.com/ai-engineer-roadmap/sitemap.xml` in Google Search
   Console and Bing Webmaster Tools.

After the redirect is live, delete or stop updating the old `/ai-roadmap/` copy so there is one version of every page.
