// See https://observablehq.com/framework/config for documentation.
export default {
  // The app’s title; used in the sidebar and webpage titles.
  title: "LLM evals - Kevin Schaul",

  // The pages and sections in the sidebar. If you don’t specify this option,
  // all pages will be listed in alphabetical order. Listing pages explicitly
  // lets you organize them into sections and have unlisted pages.
  pages: [
    {
      name: "Article tracking: Trump",
      path: "/evals/article-tracking-trump/",
    },
    {
      name: "Political fundraising emails",
      path: "/evals/political-fundraising-emails/",
    },
    {
      name: "Social media insults",
      path: "/evals/social-media-insults/",
    },
    {
      name: "NHTSA Recalls",
      path: "/evals/nhtsa-recalls/",
    },
    {
      name: "Extract FEMA incidents",
      path: "/evals/extract-fema-incidents/",
    },
    {
      name: "Grab bag",
      path: "/evals/grab-bag/",
    },
  ],

  // Content to add to the head of the page, e.g. for a favicon:
  // head: '<link rel="icon" href="observable.png" type="image/png" sizes="32x32">',

  // The path to the source root.
  root: "src",

  // Some additional configuration options and their defaults:
  // theme: "default", // try "light", "dark", "slate", etc.
  // header: "", // what to show in the header (HTML)
  footer:
    'Built by <a href="https://www.kschaul.com/">Kevin Schaul</a> with Observable.', // what to show in the footer (HTML)
  // sidebar: true, // whether to show the sidebar
  toc: false, // whether to show the table of contents
  pager: false, // whether to show previous & next links in the footer
  // output: "dist", // path to the output root for build
  // search: true, // activate search
  // linkify: true, // convert URLs in Markdown to links
  // typographer: false, // smart quotes and other typographic improvements
  // preserveExtension: false, // drop .html from URLs
  // preserveIndex: false, // drop /index from URLs
}
