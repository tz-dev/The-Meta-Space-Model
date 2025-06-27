document.addEventListener("DOMContentLoaded", function () {
  // Containers to ignore
  const ignoreContainers = ["toc", "toc_static"];

  // Function to check if a node is inside an ignored container
  function isInsideIgnoredContainer(node) {
    while (node) {
      if (node.id && ignoreContainers.includes(node.id)) return true;
      node = node.parentElement;
    }
    return false;
  }

  // Traverse text nodes (not used for replacement currently)
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
  let node;
  while (node = walker.nextNode()) {
    if (isInsideIgnoredContainer(node.parentElement)) continue;

    let replaced = false;
    let html = node.textContent;

    // Placeholder for possible text content replacements
    if (replaced) {
      const span = document.createElement("span");
      span.innerHTML = html;
      node.parentElement.replaceChild(span, node);
    }
  }

  // Add "Figure #: " before each <figcaption> in small caps (excluding ignored containers)
  const figcaptions = document.querySelectorAll("figcaption");
  let figureCounter = 1;

  figcaptions.forEach(figcaption => {
    if (isInsideIgnoredContainer(figcaption)) return;

    if (!figcaption.innerHTML.trim().startsWith("Figure")) {
      const prefix = `<span style="font-variant: small-caps;">Figure ${figureCounter++}:</span> `;
      figcaption.innerHTML = prefix + figcaption.innerHTML;
    }
  });

  // Generate Table of Contents
  function generateTOC() {
    const tocContainer = document.getElementById("toc");
    const headings = document.querySelectorAll("h2, h3");
    const baseLevel = 2;
    let currentLevel = baseLevel;
    let currentList = document.createElement("ul");
    tocContainer.appendChild(currentList);
    let listStack = [currentList];

    headings.forEach((heading, index) => {
      if (heading.textContent.trim() === "Table of Contents") return;

      const level = parseInt(heading.tagName.substring(1));
      const headingId = "heading-" + index;
      heading.setAttribute("id", headingId);

      const listItem = document.createElement("li");
      const link = document.createElement("a");
      link.href = "#" + headingId;
      link.textContent = heading.textContent;
      listItem.appendChild(link);

      if (level > currentLevel) {
        const newList = document.createElement("ul");
        listStack[listStack.length - 1].lastElementChild.appendChild(newList);
        listStack.push(newList);
        currentList = newList;
      } else if (level < currentLevel) {
        const stepsUp = currentLevel - level;
        for (let i = 0; i < stepsUp; i++) {
          listStack.pop();
        }
        currentList = listStack[listStack.length - 1];
      }

      currentList.appendChild(listItem);
      currentLevel = level;
    });
  }

  // Generate TOC on page load
  generateTOC();

  // Navigation: Set active link
  const navLinks = document.querySelectorAll("nav a");
  navLinks.forEach(link => {
    link.addEventListener("click", function () {
      // Remove all active classes
      navLinks.forEach(l => l.classList.remove("active"));
      // Set current link as active
      this.classList.add("active");
    });
  });

  // Dark Mode Toggle
  const switchCss = document.getElementById("switch_css");
  let isDarkMode = false;
  const darkModeStyle = document.createElement("style");
  darkModeStyle.id = "dark-mode-style";

  // Dark mode CSS with only color-related properties
  const darkModeCSS = `
    body {
      background: #333;
      color: #ddd;
    }
    nav h2 {
      background: #666;
    }
    nav a {
      color: #fff;
    }
    a.active {
      color: #00f0ff !important;
    }
    a {
      color: #fff;
    }
    a:hover {
      color: #00f0ff;
    }
    blockquote {
      border: 2px solid #fff;
      background: #333;
    }
    h2 {
      border-bottom: 1px solid #fff;
    }
    #toc {
      background: #222;
    }
    #toc a {
      color: #ddd;
    }
    #toc ul li a {
      background: #333;
      color: #ddd;
    }
    #toc ul li ul li a {
      background: #222;
      color: #ddd;
    }
    #toc ul li ul li a:hover, #toc ul li a:hover {
      color: #00f0ff;
    }
    table thead {
      background: #555;
      border-bottom: 2px solid #bbb;
    }
    table th {
      color: #ddd;
      background: #000 !important;
    }
    table tr:nth-child(odd) {
      background: #222!important;
    }
    table tr:nth-child(even) {
      background: #555 !important;
    }
    table caption {
      color: #999;
    }
    .diagram-container {
      border: 1px solid #666;
      background-color: #222;
    }
    .diagram-box figure {
      border: 1px solid #666;
    }
    .diagram-box figcaption {
      background-color: #000;
      color: #fff;
    }
    .description-box {
      background-color: #333;
      border-left: 1px solid #666;
    }
    .description-box h5 {
      color: #ccc;
    }
    header {
      background: #000
    }
    h1 {
      color: #ddd;
    }
    footer {
      background: #000;
      color: #999;
    }
    footer a {
      color: #000;
    }
    footer a:hover {
      color: #000;
    }
    .lightbox {
      background: rgba(255, 255, 255, 0.7);
    }
    .lightbox-content {
      background: #333;
      color: #fff;
    }
    .lightbox .close {
      color: #ccc;
    }
    nav h2 {
        background: #222;
        color: #ddd;
        padding: 3px 7px;
    }
  `;

  // Toggle dark mode
  function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    if (isDarkMode) {
      darkModeStyle.textContent = darkModeCSS;
      document.head.appendChild(darkModeStyle);
      switchCss.innerHTML = "<img src='img/bulb_off.png'>";
    } else {
      darkModeStyle.textContent = "";
      document.head.removeChild(darkModeStyle);
      switchCss.innerHTML = "<img src='img/bulb_on.png'>";
    }
  }

  // Add click event listener to toggle
  switchCss.addEventListener("click", toggleDarkMode);
});
