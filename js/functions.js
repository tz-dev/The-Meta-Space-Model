document.addEventListener("DOMContentLoaded", function () {
  // Ignorierte Container
  const ignoreContainers = ["toc", "toc_static"];

  // Funktion zur Überprüfung, ob ein Knoten in einem ignorierten Container ist
  function isInsideIgnoredContainer(node) {
    while (node) {
      if (node.id && ignoreContainers.includes(node.id)) return true;
      node = node.parentElement;
    }
    return false;
  }

  // Textknoten durchlaufen und bearbeiten
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
  let node;
  while (node = walker.nextNode()) {
    if (isInsideIgnoredContainer(node.parentElement)) continue;

    let replaced = false;
    let html = node.textContent;

    // Hier können Ersetzungen für den Textinhalt vorgenommen werden
    if (replaced) {
      const span = document.createElement("span");
      span.innerHTML = html;
      node.parentElement.replaceChild(span, node);
    }
  }

  // TOC generieren
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

  // TOC beim Laden der Seite generieren
  generateTOC();

  // Navigation: Aktiven Link setzen
  const navLinks = document.querySelectorAll("nav a");
  navLinks.forEach(link => {
    link.addEventListener("click", function () {
      // Alle aktiven Klassen entfernen
      navLinks.forEach(l => l.classList.remove("active"));

      // Aktuellen Link aktiv setzen
      this.classList.add("active");
    });
  });
});
