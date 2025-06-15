---

## 🚀 Deployment Options

### Deploy to GitHub Pages

1. Build the site:
   ```bash
   npm run build
   ```
2. Deploy to GitHub Pages:
   ```bash
   npm run deploy
   ```
3. Your site will be live at:
   `https://<your-github-username>.github.io/<repo-name>/`

### Deploy to Vercel, Netlify, or Your Own Server

1. Build the site:
   ```bash
   npm run build
   ```
2. Upload the contents of the `dist` folder to your host of choice (Vercel, Netlify, or your own web server).
   - For Vercel/Netlify: Set the build command to `npm run build` and the output directory to `dist`.
   - For custom servers: Copy the `dist` folder to your server's public directory.

> **Tip:**
> If deploying to a root domain (not a subpath), set `base: '/'` in `vite.config.js`. 