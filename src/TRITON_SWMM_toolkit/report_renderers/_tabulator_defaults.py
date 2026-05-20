"""Shared Tabulator defaults and HTML-document factory for table-grid renderers.

Centralizes:
  * CDN URL constants (Tabulator JS + CSS at a single pinned version).
  * The custom-DOM headerFilter builder JS (compound AND/OR per-column filter UI).
  * Column-spec generation with dtype-aware operator menus.
  * Tabulator options-dict factory (pagination, persistence, layout, height).
  * Self-contained HTML-document factory wrapping a Tabulator initialization.
  * JSON serialization fallback for numpy scalars / Path / pandas types.

Every renderer under report_renderers/ that emits an interactive Tabulator HTML
artifact must consume these helpers rather than duplicating the CDN constants,
HTML skeleton, or filter-builder JS. New Tabulator-emitting renderers added in
future iteration cycles inherit the compound filter UI, persistence-id
sanitization, and js-mode warning surface for free by routing through
build_html_document().

The custom-DOM headerFilter is grounded in Tabulator v6.4 documentation:
  * (cell, onRendered, success, cancel, params) editor contract — docs line 3550
    (date editor example) and 8124 (uppercase input editor example).
  * headerFilterFunc(headerValue, rowValue, rowData, filterParams) matcher
    signature — docs line 4962.
  * Nested AND/OR semantics via setFilter([{...}, [{...}, {...}]]) — docs lines
    4810-4828. The top-level array is AND-combined; nested sub-arrays are
    OR-combined.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

_TABULATOR_VERSION = "6.4.0"
_TABULATOR_JS_CDN = (
    f"https://cdn.jsdelivr.net/npm/tabulator-tables@{_TABULATOR_VERSION}"
    "/dist/js/tabulator.min.js"
)
_TABULATOR_CSS_CDN = (
    f"https://cdn.jsdelivr.net/npm/tabulator-tables@{_TABULATOR_VERSION}"
    "/dist/css/tabulator.min.css"
)

_PERSISTENCE_ID_CHARSET_RE = re.compile(r"[^A-Za-z0-9_.\-]")


# -----------------------------------------------------------------------------
# Filter-builder JS — compound per-column header filter (titleFormatter pattern)
# -----------------------------------------------------------------------------
# Architectural note (post-iter-6 pivot, 2026-05-19): the compound filter UI
# was originally implemented as a Tabulator headerFilter custom-DOM editor
# returning a wrapper <div> that body-portaled a popover. Four iterations of
# focus-management fixes failed because they targeted a non-existent
# "Tabulator auto-cancel-on-blur" mechanism (Filter.js:314 — cancel is a
# no-op stub). Tabulator's headerFilter wiring also injects three handlers
# on the returned wrapper (Filter.js:425-439, 489-493) that fight a
# popover-style UI and accumulate ghost popovers in body on
# persistence-driven editor regeneration (Filter.js:549-558 +
# Persistence.js:187-193).
#
# The pivot: detach the popover trigger entirely from Tabulator's
# headerFilter editor lifecycle by routing through titleFormatter instead.
# window.TritonReportFilter exposes three factory functions:
#
#   makeFilterTrigger(dtype)
#     Returns a Tabulator titleFormatter conforming to
#     (cell, formatterParams, onRendered) -> DOMElement. The DOM element is
#     a <span> containing the column title text + a filter-trigger button.
#     On button click, a body-portaled popover with the compound-criteria
#     builder is shown. Apply commits the JSON-stringified criteria array
#     via column.getTable().getColumn(field).setHeaderFilterValue(json) —
#     this routes through Tabulator's existing persistence + evaluation
#     paths, so headerFilterFunc still fires and persistence still
#     round-trips.
#
#   matchCriteriaList(headerValue, rowValue, rowData, filterParams)
#     The matching headerFilterFunc (UNCHANGED from prior iterations).
#     headerValue is the JSON-stringified criteria array (or empty when
#     no criteria). filterParams.dtype carries the column's declared dtype.
#     Evaluates each criterion against rowValue, combines left-to-right
#     with explicit per-criterion connector (AND / OR), no precedence.
#
#   isEmptyFilter(value)
#     The headerFilterEmptyCheck (UNCHANGED). Treats empty/zero-length
#     criteria array as "filter not active".
#
# The titleFormatter pivot makes Findings 1-4 of the 2026-05-19 root-cause
# investigation moot: there is no editor wrapper for Tabulator to attach
# click/focus/mousedown handlers to, no editor function to be re-invoked
# on persistence reload, no editor lifecycle for the popover to interact
# with. The popover is a plain DOM widget the renderer fully owns; its
# focus management is handled by native browser focus on <input>/<select>
# elements without any Tabulator interference.
#
_FILTER_BUILDER_JS = r"""
(function(global) {
  "use strict";

  // --- Operator catalogs by dtype (substrate: docs line 4720-4800) ---------
  // Each entry is [opCode, label]. opCode is the comparison key consumed by
  // evalCriterion; label is the user-visible text in the operator <select>.

  const OPS_STRING = [
    ["contains",       "contains"],
    ["not_contains",   "does not contain"],
    ["equals",         "equals"],
    ["not_equals",     "does not equal"],
    ["starts",         "starts with"],
    ["ends",           "ends with"],
    ["regex",          "matches regex"],
    ["empty",          "is empty"],
    ["not_empty",      "is not empty"],
  ];

  const OPS_NUMERIC = [
    ["eq",        "="],
    ["neq",       "≠"],
    ["lt",        "<"],
    ["lte",       "≤"],
    ["gt",        ">"],
    ["gte",       "≥"],
    ["between",   "between (a, b)"],
    ["empty",     "is empty"],
    ["not_empty", "is not empty"],
  ];

  const OPS_BOOLEAN = [
    ["true",      "is true"],
    ["false",     "is false"],
    ["empty",     "is empty"],
    ["not_empty", "is not empty"],
  ];

  const OPS_CATEGORICAL = [
    ["in",         "is one of"],
    ["not_in",     "is not one of"],
    ["equals",     "equals"],
    ["not_equals", "does not equal"],
    ["empty",      "is empty"],
    ["not_empty",  "is not empty"],
  ];

  const OP_TABLE = {
    string: OPS_STRING,
    numeric: OPS_NUMERIC,
    boolean: OPS_BOOLEAN,
    categorical: OPS_CATEGORICAL,
  };

  // Operators that don't need a value input.
  const VALUELESS_OPS = new Set(["empty", "not_empty", "true", "false"]);
  // Operators that take two values (between).
  const BINARY_OPS = new Set(["between"]);
  // Operators that take a comma-list value (in, not_in).
  const LIST_OPS = new Set(["in", "not_in"]);

  // --- Per-criterion evaluation -------------------------------------------
  // Returns boolean. dtype is "string" / "numeric" / "boolean" / "categorical".
  function evalCriterion(op, rowValue, criterionValue, dtype) {
    const isEmpty = (
      rowValue === null
      || rowValue === undefined
      || (typeof rowValue === "string" && rowValue.length === 0)
    );
    if (op === "empty")     { return isEmpty; }
    if (op === "not_empty") { return !isEmpty; }
    if (isEmpty)            { return false; }

    if (dtype === "numeric") {
      const num = Number(rowValue);
      if (Number.isNaN(num)) { return false; }
      if (op === "eq")  { return num === Number(criterionValue); }
      if (op === "neq") { return num !== Number(criterionValue); }
      if (op === "lt")  { return num <  Number(criterionValue); }
      if (op === "lte") { return num <= Number(criterionValue); }
      if (op === "gt")  { return num >  Number(criterionValue); }
      if (op === "gte") { return num >= Number(criterionValue); }
      if (op === "between") {
        const parts = String(criterionValue).split(",");
        if (parts.length !== 2) { return false; }
        const lo = Number(parts[0].trim());
        const hi = Number(parts[1].trim());
        if (Number.isNaN(lo) || Number.isNaN(hi)) { return false; }
        return num >= lo && num <= hi;
      }
      return false;
    }

    if (dtype === "boolean") {
      const truthy = (rowValue === true || rowValue === "true" || rowValue === 1);
      if (op === "true")  { return truthy; }
      if (op === "false") { return !truthy; }
      return false;
    }

    if (dtype === "categorical") {
      const sv = String(rowValue);
      if (op === "equals")     { return sv === String(criterionValue); }
      if (op === "not_equals") { return sv !== String(criterionValue); }
      if (op === "in" || op === "not_in") {
        const opts = String(criterionValue).split(",").map(function(s){ return s.trim(); });
        const inSet = opts.indexOf(sv) !== -1;
        return op === "in" ? inSet : !inSet;
      }
      return false;
    }

    // dtype === "string" (default)
    const sv = String(rowValue);
    const cv = String(criterionValue);
    if (op === "contains")     { return sv.toLowerCase().indexOf(cv.toLowerCase()) !== -1; }
    if (op === "not_contains") { return sv.toLowerCase().indexOf(cv.toLowerCase()) === -1; }
    if (op === "equals")       { return sv === cv; }
    if (op === "not_equals")   { return sv !== cv; }
    if (op === "starts")       { return sv.toLowerCase().indexOf(cv.toLowerCase()) === 0; }
    if (op === "ends")         {
      return sv.toLowerCase().lastIndexOf(cv.toLowerCase()) === sv.length - cv.length;
    }
    if (op === "regex") {
      try {
        return new RegExp(cv).test(sv);
      } catch (e) {
        return false;
      }
    }
    return false;
  }

  // --- In-column criteria-list evaluator (headerFilterFunc) ---------------
  // headerValue is the JSON-stringified criteria array (set by success() in
  // the builder UI). filterParams.dtype is the column's declared dtype.
  function matchCriteriaList(headerValue, rowValue, rowData, filterParams) {
    if (!headerValue) { return true; }
    let criteria;
    try {
      criteria = JSON.parse(headerValue);
    } catch (e) {
      return true;
    }
    if (!Array.isArray(criteria) || criteria.length === 0) { return true; }
    const dtype = (filterParams && filterParams.dtype) || "string";
    let acc = evalCriterion(criteria[0].op, rowValue, criteria[0].value, dtype);
    for (let i = 1; i < criteria.length; i++) {
      const c = criteria[i];
      const r = evalCriterion(c.op, rowValue, c.value, dtype);
      if (c.connector === "OR") { acc = acc || r; }
      else                       { acc = acc && r; }
    }
    return acc;
  }

  // Tabulator's headerFilterEmptyCheck. Treats a JSON string of [] or "" as
  // empty so an unconfigured filter passes every row (docs line 4944).
  function isEmptyFilter(value) {
    if (!value) { return true; }
    try {
      const arr = JSON.parse(value);
      return !Array.isArray(arr) || arr.length === 0;
    } catch (e) {
      return true;
    }
  }

  // --- Builder UI (titleFormatter trigger + independent body popover) -----
  // The titleFormatter contract: (cell, formatterParams, onRendered) ->
  // DOMElement or string. Tabulator places the returned element inside the
  // column title cell with NO injected click/focus/mousedown listeners
  // (cf. Filter.js:425-498 which DOES inject such listeners on
  // headerFilter editors). The popover is owned entirely by our code; its
  // value-input typing is unobstructed by Tabulator's editor lifecycle.
  //
  // State commit: Apply calls col.setHeaderFilterValue(json) which routes
  // through Tabulator's existing setHeaderFilterValue path (Filter.js:550)
  // — the headerFilterFunc (TritonReportFilter.matchCriteriaList) and the
  // persistence subsystem both still fire. Caveat: setHeaderFilterValue
  // calls generateHeaderFilterElement(column, value, true), but with NO
  // headerFilter declared on the column (see VMS-3 below) that path is a
  // no-op because the column has no editor function registered.
  function makeFilterTrigger(dtype, columnTitle, columnField) {
    return function(cell, formatterParams, onRendered) {
      // iter 8 — `display: flex` (block-level) instead of `inline-flex`.
      // `.tabulator-col-title` is a block-level <div> with white-space:
      // nowrap + overflow: hidden + text-overflow: ellipsis. An
      // `inline-flex` container nested inside an inline rendering context
      // with `flex: 1` children can collapse to zero intrinsic width when
      // the flex item's `flex-basis: 0` interacts with the parent's
      // overflow:hidden semantics — observed in iter 7 as titleSpan text
      // not rendering. `display: flex` is block-level: the container
      // fills the parent's content box vertically and honors `width:
      // 100%` horizontally. `min-height: 100%` matches the data-row
      // height (Tabulator's column-header row otherwise collapses to the
      // intrinsic height of the formatter output — observed in iter 7 as
      // a vertically-short header row vs data rows; the screenshot's red
      // annotation at the column-header-vs-data height differential
      // pinned this).
      const container = document.createElement("span");
      container.className = "trf-title-container";
      container.style.display = "flex";
      container.style.alignItems = "center";
      container.style.gap = "4px";
      container.style.width = "100%";
      container.style.minHeight = "100%";

      const titleSpan = document.createElement("span");
      titleSpan.className = "trf-title-text";
      titleSpan.textContent = columnTitle;
      titleSpan.style.flex = "1 1 auto";
      titleSpan.style.minWidth = "0";
      titleSpan.style.overflow = "hidden";
      titleSpan.style.textOverflow = "ellipsis";
      titleSpan.style.whiteSpace = "nowrap";
      container.appendChild(titleSpan);

      const triggerBtn = document.createElement("button");
      triggerBtn.type = "button";
      triggerBtn.className = "trf-filter-trigger";
      triggerBtn.setAttribute("aria-haspopup", "dialog");
      triggerBtn.setAttribute("aria-label", "Open compound filter for " + columnTitle);
      triggerBtn.textContent = "▾";
      triggerBtn.style.padding = "0 4px";
      triggerBtn.style.cursor = "pointer";
      triggerBtn.style.border = "1px solid #888";
      triggerBtn.style.background = "#fff";
      triggerBtn.style.fontSize = "10px";
      container.appendChild(triggerBtn);

      const statusBadge = document.createElement("span");
      statusBadge.className = "trf-filter-status";
      statusBadge.setAttribute("aria-live", "polite");
      statusBadge.style.fontSize = "10px";
      statusBadge.style.color = "#2E7D32";
      statusBadge.style.marginLeft = "2px";
      container.appendChild(statusBadge);

      const popover = document.createElement("div");
      popover.className = "trf-filter-popover";
      popover.setAttribute("role", "dialog");
      popover.setAttribute("aria-label", "Build compound filter for " + columnTitle);
      popover.style.position = "fixed";
      popover.style.zIndex = "5000";
      popover.style.background = "#fff";
      popover.style.border = "1px solid #888";
      popover.style.padding = "6px";
      popover.style.minWidth = "260px";
      popover.style.boxShadow = "0 4px 12px rgba(0,0,0,0.25)";
      popover.style.display = "none";
      document.body.appendChild(popover);

      // iter 9.4 — Native HTML <datalist> for filter value autocomplete.
      // The datalist is referenced from each criterion-row value <input>
      // via the `list` attribute; browser handles filter-as-you-type,
      // arrow-key navigation, click-to-select, and Enter-to-select for
      // free. No custom JS needed for the dropdown UX.
      // Populated lazily on first openPopover() (and refreshed on each
      // open thereafter, since the active row set may have changed due
      // to filtering / pagination / clipboard activity). Suggestions are
      // sourced from cell.getTable().getColumn(field).getCells() —
      // ALL rows in the data set, not just the visible page.
      const datalistId = "trf-dl-" + columnField.replace(/[^a-zA-Z0-9_]/g, "_");
      const datalist = document.createElement("datalist");
      datalist.id = datalistId;
      popover.appendChild(datalist);
      function populateDatalist() {
        try {
          const table = cell.getTable();
          const col = table.getColumn(columnField);
          if (!col) { return; }
          const seen = new Set();
          const values = [];
          col.getCells().forEach(function(c) {
            const v = c.getValue();
            if (v === null || v === undefined) { return; }
            const s = String(v);
            if (s.length === 0) { return; }
            if (seen.has(s)) { return; }
            seen.add(s);
            values.push(s);
          });
          values.sort();
          while (datalist.firstChild) {
            datalist.removeChild(datalist.firstChild);
          }
          values.forEach(function(v) {
            const opt = document.createElement("option");
            opt.value = v;
            datalist.appendChild(opt);
          });
        } catch (e) {
          if (window.console && window.console.warn) {
            window.console.warn("trf datalist populate failed for " + columnField, e);
          }
        }
      }

      const criteriaList = document.createElement("div");
      criteriaList.className = "trf-criteria-list";
      popover.appendChild(criteriaList);

      const addBtn = document.createElement("button");
      addBtn.type = "button";
      addBtn.textContent = "+ Add criterion";
      addBtn.className = "trf-add-btn";
      addBtn.style.marginTop = "4px";
      popover.appendChild(addBtn);

      const footer = document.createElement("div");
      footer.style.marginTop = "6px";
      footer.style.textAlign = "right";
      const clearBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.textContent = "Clear";
      clearBtn.className = "trf-clear-btn";
      clearBtn.style.marginLeft = "4px";
      const applyBtn = document.createElement("button");
      applyBtn.type = "button";
      applyBtn.textContent = "Apply";
      applyBtn.className = "trf-apply-btn";
      applyBtn.style.marginLeft = "4px";
      footer.appendChild(clearBtn);
      footer.appendChild(applyBtn);
      popover.appendChild(footer);

      // closure state -- array of {op, value, connector} records.
      // Initial state seeded from Tabulator's persisted header-filter value
      // on first render via cell.getColumn().getHeaderFilterValue().
      let criteria = [];
      try {
        const col = cell.getColumn();
        const existing = col && col.getHeaderFilterValue ? col.getHeaderFilterValue() : "";
        if (existing) {
          const parsed = JSON.parse(existing);
          if (Array.isArray(parsed)) { criteria = parsed; }
        }
      } catch (e) { criteria = []; }

      function rebuildList() {
        while (criteriaList.firstChild) {
          criteriaList.removeChild(criteriaList.firstChild);
        }
        criteria.forEach(function(c, idx) {
          const row = document.createElement("div");
          row.className = "trf-criteria-row";
          row.style.display = "flex";
          row.style.gap = "4px";
          row.style.marginBottom = "3px";
          row.style.alignItems = "center";

          if (idx > 0) {
            const conn = document.createElement("select");
            conn.className = "trf-connector";
            conn.setAttribute("aria-label", "Connector");
            ["AND", "OR"].forEach(function(label) {
              const opt = document.createElement("option");
              opt.value = label;
              opt.textContent = label;
              if (c.connector === label) { opt.selected = true; }
              conn.appendChild(opt);
            });
            conn.addEventListener("change", function() {
              criteria[idx].connector = conn.value;
            });
            row.appendChild(conn);
          } else {
            const spacer = document.createElement("span");
            spacer.style.display = "inline-block";
            spacer.style.width = "44px";
            row.appendChild(spacer);
          }

          const opSel = document.createElement("select");
          opSel.className = "trf-operator";
          opSel.setAttribute("aria-label", "Operator");
          (OP_TABLE[dtype] || OPS_STRING).forEach(function(pair) {
            const opt = document.createElement("option");
            opt.value = pair[0];
            opt.textContent = pair[1];
            if (c.op === pair[0]) { opt.selected = true; }
            opSel.appendChild(opt);
          });
          row.appendChild(opSel);

          const valInput = document.createElement("input");
          valInput.type = "text";
          valInput.className = "trf-value";
          valInput.setAttribute("aria-label", "Value");
          // iter 9.4 — Reference the column's <datalist> for native
          // browser autocomplete (filter-as-you-type + arrow keys +
          // click/Enter to select).
          valInput.setAttribute("list", datalistId);
          valInput.value = (c.value === undefined || c.value === null) ? "" : String(c.value);
          valInput.style.flex = "1";
          if (BINARY_OPS.has(c.op)) {
            valInput.placeholder = "min, max";
          } else if (LIST_OPS.has(c.op)) {
            valInput.placeholder = "a, b, c";
          } else if (VALUELESS_OPS.has(c.op)) {
            valInput.style.visibility = "hidden";
          } else {
            valInput.placeholder = "value";
          }
          row.appendChild(valInput);

          opSel.addEventListener("change", function() {
            criteria[idx].op = opSel.value;
            if (VALUELESS_OPS.has(opSel.value)) {
              valInput.style.visibility = "hidden";
              valInput.value = "";
            } else {
              valInput.style.visibility = "visible";
              if (BINARY_OPS.has(opSel.value))   { valInput.placeholder = "min, max"; }
              else if (LIST_OPS.has(opSel.value)) { valInput.placeholder = "a, b, c"; }
              else                                { valInput.placeholder = "value"; }
            }
          });
          valInput.addEventListener("input", function() {
            criteria[idx].value = valInput.value;
          });

          const rmBtn = document.createElement("button");
          rmBtn.type = "button";
          rmBtn.textContent = "×";
          rmBtn.className = "trf-remove";
          rmBtn.setAttribute("aria-label", "Remove criterion");
          rmBtn.style.marginLeft = "4px";
          rmBtn.addEventListener("click", function() {
            criteria.splice(idx, 1);
            rebuildList();
          });
          row.appendChild(rmBtn);

          criteriaList.appendChild(row);
        });
      }

      function updateStatusBadge() {
        if (criteria.length === 0) {
          statusBadge.textContent = "";
          triggerBtn.style.background = "#fff";
        } else {
          statusBadge.textContent = "(" + criteria.length + ")";
          triggerBtn.style.background = "#E8F0F8";
        }
      }

      function positionPopover() {
        const rect = triggerBtn.getBoundingClientRect();
        const popWidth = Math.max(popover.offsetWidth, 260);
        const maxLeft = window.innerWidth - popWidth - 8;
        const left = Math.max(8, Math.min(rect.left, maxLeft));
        popover.style.left = left + "px";
        popover.style.top = (rect.bottom + 2) + "px";
      }
      function openPopover() {
        // iter 9.4 — Refresh the column's <datalist> suggestions on each
        // open. The active row set may have changed since the popover
        // was last opened (other column filters, pagination, etc.), so
        // we re-collect unique values from the current Tabulator state.
        populateDatalist();
        popover.style.display = "block";
        positionPopover();
        const firstSel = popover.querySelector(".trf-operator");
        if (firstSel) {
          firstSel.focus();
        } else {
          addBtn.focus();
        }
      }
      function closePopover() {
        popover.style.display = "none";
        triggerBtn.focus();
      }
      function repositionIfOpen() {
        if (popover.style.display === "block") { positionPopover(); }
      }
      window.addEventListener("scroll", repositionIfOpen, true);
      window.addEventListener("resize", repositionIfOpen);

      addBtn.addEventListener("click", function(ev) {
        ev.stopPropagation();
        const defaultOp = ((OP_TABLE[dtype] || OPS_STRING)[0] || ["contains"])[0];
        criteria.push({
          op: defaultOp,
          value: "",
          connector: criteria.length === 0 ? null : "AND",
        });
        rebuildList();
        const newRow = criteriaList.lastChild;
        if (newRow) {
          const newValInput = newRow.querySelector(".trf-value");
          if (newValInput && newValInput.style.visibility !== "hidden") {
            newValInput.focus();
          }
        }
      });

      // iter 9.5 — Per-column active filter function reference. Tracks
      // the function passed to table.addFilter so we can remove it on
      // Apply (before adding a new one) and on Clear. Each column owns
      // its own closure variable; multiple columns can have filters
      // simultaneously and Tabulator ANDs them.
      let activeFilterFunc = null;

      function applyFilter(criteriaPayload) {
        // Root-cause-of-iter-9.4-bug fix (iter 9.5): the prior
        // implementation called col.setHeaderFilterValue(payload), which
        // routes through Tabulator's headerFilter editor pipeline at
        // Filter.js:549-558. That pipeline requires
        // `column.modules.filter` to be initialized, which only happens
        // when `def.headerFilter` is truthy at Filter.js:194-200. With
        // our iter-7 VMS-3 architectural pivot (headerFilter: false),
        // column.modules.filter is NEVER set up, so setHeaderFilterValue
        // hits the `if(column.modules.filter && ...filter.headerElement)`
        // gate and silently no-ops (Tabulator emits a console.warn but
        // the filter is never registered). The (1) status badge updated
        // (renderer-local UI state) but no row filtering happened.
        //
        // Fix: switch to table.addFilter / table.removeFilter (the
        // general-filter pipeline), which works regardless of
        // headerFilter setting. Each column maintains its own closure
        // reference to the active filter function; Apply removes the
        // prior + adds the new; Clear removes only.
        const table = cell.getTable();
        if (activeFilterFunc) {
          try { table.removeFilter(activeFilterFunc); } catch (e) {}
          activeFilterFunc = null;
        }
        if (criteriaPayload !== "") {
          activeFilterFunc = function(rowData) {
            try {
              const fieldVal = rowData[columnField];
              return TritonReportFilter.matchCriteriaList(
                criteriaPayload, fieldVal, rowData, {dtype: dtype}
              );
            } catch (e) {
              return true;  // fail-open: don't drop rows on matcher error
            }
          };
          try {
            table.addFilter(activeFilterFunc);
          } catch (e) {
            if (window.console && window.console.warn) {
              window.console.warn("trf: addFilter failed for " + columnField, e);
            }
            activeFilterFunc = null;
          }
        }
      }

      applyBtn.addEventListener("click", function(ev) {
        ev.stopPropagation();
        const payload = criteria.length === 0 ? "" : JSON.stringify(criteria);
        applyFilter(payload);
        updateStatusBadge();
        closePopover();
      });

      clearBtn.addEventListener("click", function(ev) {
        ev.stopPropagation();
        criteria = [];
        rebuildList();
        applyFilter("");
        updateStatusBadge();
        closePopover();
      });

      triggerBtn.addEventListener("click", function(ev) {
        ev.stopPropagation();
        if (popover.style.display === "block") { closePopover(); } else { openPopover(); }
      });
      triggerBtn.addEventListener("keydown", function(ev) {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          openPopover();
        }
      });
      popover.addEventListener("keydown", function(ev) {
        if (ev.key === "Escape") {
          ev.preventDefault();
          closePopover();
        }
      });

      // iter 8 — Switched from `click` (capture) to `mousedown` (bubble).
      // The `click` event fires on mouseup; for a drag from the trigger
      // button into the popover (or anywhere else), the click event's
      // target is the COMMON ANCESTOR of mousedown-target and
      // mouseup-target (often <body>). `popover.contains(body) === false`
      // -> closePopover incorrectly fires, dropping focus from a
      // popover-internal <input> mid-typing. `mousedown` fires on press
      // (not on release), so an outside-popover mousedown closes the
      // popover, but a drag-from-trigger-then-release-inside-popover does
      // NOT trigger close. Combined with the popover's own
      // mousedown/mouseup stopPropagation listeners below, no
      // document-level handler ever sees a popover-internal mouse event.
      //
      // iter 9.3 — Early-return when popover is closed. With 39 columns,
      // 39 onDocMousedown listeners are registered on document. On EVERY
      // mousedown anywhere on the page (including white space, sidebar
      // buttons, table data cells, scrollbar, etc.), ALL 39 listeners
      // fire. Without this guard, each one called closePopover ->
      // triggerBtn.focus() unconditionally. The result was 39 focus
      // calls per click; the LAST focus won, parking focus on one
      // column's <button> per click. That focused button's intrinsic
      // width grew by the browser's focus outline (~2px), making that
      // column's HEADER cell wider than its DATA cells (which have no
      // focusable button), persistently shifting the column-header row
      // out of alignment with the data column. The early-return below
      // ensures closePopover is only called by the listener whose
      // popover is actually open (typically 0 or 1 listener at a time),
      // eliminating the focus storm.
      function onDocMousedown(ev) {
        if (popover.style.display !== "block") { return; }
        if (!container.contains(ev.target) && !popover.contains(ev.target)) {
          closePopover();
        }
      }
      document.addEventListener("mousedown", onDocMousedown);

      // Stop popover-internal mouse events from bubbling to document.
      // Defense-in-depth against any future document-level handler that
      // might mis-handle popover events. Stops mousedown, mouseup, AND
      // click — all three event types that could otherwise leak to
      // ancestors and trigger blur/close/clear-selection side effects.
      popover.addEventListener("mousedown", function(ev) { ev.stopPropagation(); });
      popover.addEventListener("mouseup",   function(ev) { ev.stopPropagation(); });
      popover.addEventListener("click",     function(ev) { ev.stopPropagation(); });

      // Initial render once attached.
      if (onRendered) {
        onRendered(function() {
          rebuildList();
          updateStatusBadge();
        });
      } else {
        rebuildList();
        updateStatusBadge();
      }

      return container;
    };
  }

  global.TritonReportFilter = {
    makeFilterTrigger: makeFilterTrigger,
    matchCriteriaList: matchCriteriaList,
    isEmptyFilter: isEmptyFilter,
  };
})(window);
"""


# -----------------------------------------------------------------------------
# Python factories
# -----------------------------------------------------------------------------

def sanitize_persistence_id(value: str) -> str:
    """Sanitize a string to Tabulator persistenceID-safe charset.

    Tabulator's persistenceID is used as a localStorage key. The
    TableInteractiveConfig.persistence_id field validator enforces
    ``^[A-Za-z0-9_.\\-]+$`` so user-supplied IDs are charset-safe at
    config-load time. When the renderer derives the ID from
    analysis.analysis_id at render time, the analysis ID may contain
    spaces, slashes, or other characters that fail the charset — this
    helper munges those into ``_``. Empty input collapses to ``_``.
    """
    sanitized = _PERSISTENCE_ID_CHARSET_RE.sub("_", value or "")
    return sanitized or "_"


def dtype_for_dataframe_column(series: pd.Series) -> str:
    """Map a pandas Series dtype to a Tabulator filter dtype label.

    Returns one of ``"numeric"``, ``"boolean"``, ``"categorical"``, or
    ``"string"`` — the four labels recognized by ``OP_TABLE`` inside
    ``_FILTER_BUILDER_JS``. The mapping is intentionally narrow so the
    filter UI's operator menus stay predictable.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    return "string"


def build_columns_spec(
    df: pd.DataFrame,
    *,
    visible_columns_default: Iterable[str] | None,
    header_filter: bool,
) -> list[dict]:
    """Build the Tabulator ``columns`` array from a DataFrame.

    When ``header_filter`` is True, each column gets a compound-filter UI
    via a ``titleFormatter`` that renders the column title + a filter-trigger
    button (``TritonReportFilter.makeFilterTrigger``). The matching
    ``headerFilterFunc`` (``TritonReportFilter.matchCriteriaList``), the
    per-column ``headerFilterFuncParams`` dict carrying the column's
    dtype, and the ``headerFilterEmptyCheck`` (``TritonReportFilter.isEmptyFilter``)
    are retained so Tabulator's filter evaluation + persistence subsystems
    continue to fire. The architectural pivot (2026-05-19): ``headerFilter``
    itself is declared ``False`` so Tabulator does NOT instantiate an editor
    wrapper — Filter.js:425-498 wiring is bypassed entirely. The trigger
    button and body-portaled popover are owned by the titleFormatter and
    commit filter state via ``cell.getColumn().setHeaderFilterValue(json)``.
    """
    visible_set = set(visible_columns_default) if visible_columns_default is not None else None
    spec: list[dict] = []
    for col in df.columns:
        col_str = str(col)
        col_spec: dict = {"title": col_str, "field": col_str}
        if header_filter:
            dtype = dtype_for_dataframe_column(df[col])
            # Sentinel strings here are replaced by literal JS expressions in
            # build_html_document() so the rendered JS calls into
            # window.TritonReportFilter rather than reading a string.
            col_spec["titleFormatter"] = f"__TRF_FILTER_TRIGGER__{dtype}__{col_str}__"
            col_spec["headerFilter"] = False
            col_spec["headerFilterFunc"] = "__TRF_MATCHER__"
            col_spec["headerFilterFuncParams"] = {"dtype": dtype}
            col_spec["headerFilterEmptyCheck"] = "__TRF_EMPTY_CHECK__"
        if visible_set is not None:
            col_spec["visible"] = col_str in visible_set
        spec.append(col_spec)
    return spec


def build_options_dict(
    df: pd.DataFrame,
    *,
    columns_spec: list[dict],
    table_height: str,
    pagination_size: int,
    persistence_id: str | None,
    extra_options: Mapping[str, Any] | None = None,
) -> dict:
    """Assemble the Tabulator constructor options dict from configured pieces.

    ``persistence_id``, when provided, is wired to Tabulator's
    ``persistence:true`` + ``persistenceID`` and also to the
    headerFilter persistence key. ``pagination_size <= 0`` disables
    pagination; the table falls back to virtual-scroll (which requires
    ``table_height`` per tabulator-architecture: "tables MUST have height
    set or virtual DOM disengages"). ``extra_options`` are merged last
    and may override any default; callers use this for renderer-specific
    additions (e.g., ``columnDefaults`` for per-column option overrides).
    """
    options: dict = {
        "data": df.to_dict(orient="records"),
        "columns": columns_spec,
        # fitDataStretch: columns size to content; user scrolls horizontally
        # when total width exceeds viewport. Used in place of fitColumns
        # because fitColumns divides viewport across N columns -> with N=39
        # each column collapses to ~30-50px, headers truncate to single chars,
        # and filter inputs become unreadable (iter 1v2 user feedback at
        # scratch L3997-4002 + annotated screenshot showed this failure).
        "layout": "fitDataStretch",
        "height": table_height,
        # iter 8 — Row-selection DISABLED (selectable: False) to remove
        # SelectRow.js's row-click handler chain. With selectable:1 (iter 5
        # state), every row-click invoked self.table._clearSelection()
        # (SelectRow.js:112-120 -> Tabulator.js:139-154) which called
        # window.getSelection().removeAllRanges(). Empirically pinned by
        # user iter-7 testing: drag-selecting text in a single-row cell
        # (e.g., scenario_directory path) and releasing the mouse fires a
        # `click` event on the row -> _clearSelection -> highlight cleared.
        # selectable:False removes the entire click+mousedown+mouseenter
        # chain at SelectRow.js:108-156, restoring native browser
        # text-selection behavior. clipboardCopyRowRange:"selected" falls
        # back to "active" (full filtered rowset) when no rows are
        # selectable, matching the existing "Copy table" semantics —
        # confirmed unchanged from iter 5 behavior. clipboard:"copy"
        # (paste disabled) is retained for Ctrl+C; clipboardCopyConfig
        # omits headers per user spec at scratch L4260.
        "clipboard": "copy",
        "clipboardCopyRowRange": "active",
        "clipboardCopyConfig": {
            "columnHeaders": False,
            "columnGroups": False,
            "rowGroups": False,
            "columnCalcs": False,
            "dataTree": False,
        },
        "selectable": False,
        # iter 9.1 — `resizableColumns: False` removes the column-resize
        # handle (a 3px-wide <span class="tabulator-col-resize-handle">
        # positioned at the right edge of each column header via
        # ResizeColumns.js:144-157). The handle binds its own mousedown +
        # click listeners and starts a column-resize drag on press. With
        # 39 columns at fitDataStretch sizing, the right-edge handle is
        # adjacent to the next column's left edge and is easy to click by
        # accident when aiming at a column-title-area target. Combined
        # with `persistence.columns: False` below, this eliminates the
        # "click → column-width latched change → header/body misalign"
        # path observed in iter 9 after VMS-1 disabled headerSort.
        "resizableColumns": False,
    }
    if pagination_size > 0:
        options["pagination"] = "local"
        options["paginationSize"] = pagination_size
    if persistence_id is not None:
        # iter 9.1 — `persistence.columns: False` (was True in iter 5+).
        # Persistence.js:154-156 saves the current column widths to
        # localStorage on every `column-resized`, `column-width`, AND
        # `layout-refreshed` event. The `layout-refreshed` subscription
        # is the destabilizer: any layout pass (including the implicit
        # ones triggered by clicks that touch column-width state) fires
        # the save. Combined with the save-on-reload restoration path,
        # this creates a feedback loop where column-width state can latch
        # into one of two stable configurations that alternate per click.
        # Sort/filter/headerFilter/group/page persistence remain enabled
        # because those are not column-width-coupled and the user wants
        # filter state to round-trip across reloads.
        options["persistence"] = {
            "sort": True,
            "filter": True,
            "headerFilter": True,
            "group": True,
            "page": True,
            "columns": False,
        }
        options["persistenceID"] = persistence_id
    if extra_options:
        for k, v in extra_options.items():
            options[k] = v
    return options


def build_html_document(
    *,
    title: str,
    container_id: str,
    body_heading_html: str,
    options: dict,
    js_mode: str = "cdn",
    renderer_name: str = "<unknown>",
    column_groups: list[tuple[str, list[str], str | None]] | None = None,
) -> str:
    """Build a self-contained Tabulator HTML document.

    The options dict is JSON-serialized with sentinel-string substitution:
    ``__TRF_FILTER_TRIGGER__<dtype>__<field>__`` -> JS expression invoking
    ``TritonReportFilter.makeFilterTrigger("<dtype>", "<field>", "<field>")``;
    ``__TRF_MATCHER__`` -> ``TritonReportFilter.matchCriteriaList``;
    ``__TRF_EMPTY_CHECK__`` -> ``TritonReportFilter.isEmptyFilter``. This
    pattern is borrowed from Plotly's similar trick of carrying JS
    callables through JSON via string-replace.

    ``column_groups``, when provided, drives the sidebar's group layout:
    each tuple is ``(group_label, column_field_list, optional_footnote)``.
    Columns not in any group fall under an auto-generated "Other"
    catchall group. When ``column_groups`` is None, the sidebar renders
    a flat checkbox list with no subheaders.

    ``js_mode="inline"`` is not yet implemented; emits a one-time warning
    and falls back to CDN. Tracked at /design-figure Phase C follow-up.
    """
    if js_mode == "inline":
        warnings.warn(
            f"{renderer_name}: tabulator_js_mode='inline' is not yet "
            "implemented; falling back to CDN. Inline bundling is "
            "scheduled for /design-figure Phase C iteration.",
            stacklevel=2,
        )

    options_json = json.dumps(options, default=_json_default)
    # Sentinel substitution — placed AFTER json.dumps so the sentinels are not
    # quoted as bare identifiers; the regex matches the JSON-stringified form.
    # The titleFormatter sentinel carries both the dtype AND the column field
    # name (used as the column-title argument to makeFilterTrigger); since
    # JSON keys preserve string content with backslash-escaping but field
    # names that contain underscores or dots are common, the regex uses a
    # non-greedy capture for the field segment terminated by the trailing
    # `__"`. The substitution emits a JS expression with the field name
    # quoted twice (once as the title, once as the field).
    options_json = re.sub(
        r'"__TRF_FILTER_TRIGGER__([a-z]+)__(.+?)__"',
        r'TritonReportFilter.makeFilterTrigger("\1", "\2", "\2")',
        options_json,
    )
    options_json = options_json.replace(
        '"__TRF_MATCHER__"', "TritonReportFilter.matchCriteriaList",
    )
    options_json = options_json.replace(
        '"__TRF_EMPTY_CHECK__"', "TritonReportFilter.isEmptyFilter",
    )

    head_assets = (
        f'<link rel="stylesheet" href="{_TABULATOR_CSS_CDN}">\n'
        f'<script src="{_TABULATOR_JS_CDN}"></script>'
    )

    sidebar_id = f"{container_id}-sidebar"
    layout_id = f"{container_id}-layout"

    # Serialize column_groups (if provided) as a JS literal embedded in the
    # rendered HTML. Each entry is {label, columns, footnote}. The sidebar's
    # refresh() iterates this structure instead of getColumns() so columns
    # render under their semantic-provenance group headings.
    if column_groups:
        groups_payload = [
            {"label": label, "columns": list(columns), "footnote": footnote}
            for (label, columns, footnote) in column_groups
        ]
    else:
        groups_payload = []
    column_groups_json = json.dumps(groups_payload)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"{head_assets}\n"
        "<style>\n"
        "body { margin: 0; padding: 12px; "
        'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", '
        "Roboto, sans-serif; }\n"
        "h2 { margin-top: 0; }\n"
        # Two-column layout: column-visibility sidebar on left + table on right.
        # The wrapper uses flex with min-width:0 on the table column so the
        # table can shrink + scroll horizontally without overflowing the body.
        f"#{layout_id} {{ display: flex; gap: 12px; align-items: flex-start; }}\n"
        f"#{sidebar_id} {{ flex: 0 0 220px; max-height: 75vh; "
        "overflow-y: auto; border: 1px solid #DADADA; padding: 8px; "
        "font-size: 12px; background: #FAFAFA; }\n"
        f"#{sidebar_id} h3 {{ margin: 0 0 6px 0; font-size: 13px; "
        "color: #232D4B; }\n"
        f"#{sidebar_id} h4 {{ margin: 10px 0 4px 0; font-size: 12px; "
        "color: #232D4B; border-bottom: 1px solid #DADADA; "
        "padding-bottom: 2px; }\n"
        f"#{sidebar_id} label {{ display: block; padding: 2px 0; "
        "cursor: pointer; user-select: none; }\n"
        f"#{sidebar_id} input[type=checkbox] {{ margin-right: 6px; }}\n"
        f"#{sidebar_id} .trf-toggle-all, "
        f"#{sidebar_id} .trf-copy-table {{ font-size: 11px; padding: 2px 6px; "
        "margin-bottom: 6px; margin-right: 4px; cursor: pointer; }\n"
        # iter 8 — per-group toggle buttons (.trf-group-toggle) are
        # styled smaller and lighter than the global .trf-toggle-all
        # buttons to keep visual emphasis on the global controls. They
        # share the cursor: pointer + border: 1px solid #BBB hairline.
        f"#{sidebar_id} .trf-group-toggle {{ font-size: 10px; padding: 1px 4px; "
        "margin-right: 3px; cursor: pointer; border: 1px solid #BBB; "
        "background: #fff; color: #232D4B; }\n"
        f"#{sidebar_id} .trf-group-toggle:hover {{ background: #E8F0F8; }}\n"
        f"#{sidebar_id} .trf-copy-table {{ background: #E8F0F8; "
        "border: 1px solid #888; }\n"
        # iter 9.2 — Reset all button (clears localStorage persistence
        # keys + reloads). Styled with a muted warning color (amber-tinted
        # background + darker border) to signal a state-clearing action
        # without being alarming. Same size as .trf-copy-table for visual
        # parity with the other sidebar-header global buttons.
        f"#{sidebar_id} .trf-reset-all {{ font-size: 11px; padding: 2px 6px; "
        "margin-bottom: 6px; margin-right: 4px; cursor: pointer; "
        "background: #FFF3E0; border: 1px solid #B26A00; color: #5D2E00; }\n"
        f"#{sidebar_id} .trf-reset-all:hover {{ background: #FFE0B2; }}\n"
        f"#{sidebar_id} .trf-copy-status {{ font-size: 11px; color: #2E7D32; "
        "min-height: 14px; margin-bottom: 4px; }\n"
        f"#{sidebar_id} .trf-group-footnote {{ font-size: 10px; "
        "color: #666; font-style: italic; margin: 2px 0 6px 0; "
        "line-height: 1.3; }\n"
        f"#{container_id} {{ flex: 1 1 auto; min-width: 0; }}\n"
        # Filter-popover clipping fix: Tabulator's column-header cells
        # default to overflow:hidden, which clips the absolute-positioned
        # .trf-filter-popover below the header. Setting overflow:visible
        # on the header containers and giving the column header a higher
        # stacking context lets the popover escape (per iter 1v2 user
        # feedback at scratch L4001-4002 + annotated screenshot).
        ".tabulator-header, .tabulator-headers, .tabulator-col, "
        ".tabulator-col-content { overflow: visible !important; }\n"
        ".tabulator-col.tabulator-col-active { z-index: 100; }\n"
        ".trf-filter-popover { z-index: 2000 !important; }\n"
        ".trf-filter-popover button { padding: 2px 8px; cursor: pointer; }\n"
        ".trf-criteria-row select, .trf-criteria-row input { padding: 2px 4px; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_heading_html}"
        f'<div id="{layout_id}">\n'
        f'  <aside id="{sidebar_id}" role="region" aria-label="Column visibility">\n'
        "    <h3>Columns</h3>\n"
        '    <button type="button" class="trf-toggle-all" data-action="show-all">Show all</button>\n'
        '    <button type="button" class="trf-toggle-all" data-action="hide-all">Hide all</button>\n'
        '    <button type="button" class="trf-copy-table" '
        'title="Copy filtered rows × visible columns to clipboard (tab-separated). '
        'Paste into a spreadsheet to keep cell boundaries.">Copy table</button>\n'
        '    <button type="button" class="trf-reset-all" '
        'title="Clear all persisted table state (sort / filter / headerFilter / group / page / '
        'cached column widths) from localStorage and reload the page. '
        'Use this if the table behaves oddly after a config change.">Reset all</button>\n'
        '    <div class="trf-copy-status" aria-live="polite" role="status"></div>\n'
        '    <div class="trf-column-list"></div>\n'
        "  </aside>\n"
        f'  <div id="{container_id}"></div>\n'
        "</div>\n"
        "<script>\n"
        f"{_FILTER_BUILDER_JS}\n"
        f"const tableOptions = {options_json};\n"
        f'const __trfTable = new Tabulator("#{container_id}", tableOptions);\n'
        "// Column-visibility sidebar — populated after tableBuilt so\n"
        "// column instances are addressable. Each row is a checkbox bound\n"
        "// to column.show()/column.hide(). Tabulator's persistence:columns\n"
        "// option (set in build_options_dict) round-trips the visible state\n"
        "// to localStorage, so the sidebar checkboxes re-hydrate on reload.\n"
        f"const __trfColumnGroups = {column_groups_json};\n"
        '__trfTable.on("tableBuilt", function() {\n'
        f'  const sidebar = document.getElementById("{sidebar_id}");\n'
        '  const list = sidebar.querySelector(".trf-column-list");\n'
        "\n"
        "  function makeCheckboxRow(col) {\n"
        '    const label = document.createElement("label");\n'
        '    const cb = document.createElement("input");\n'
        '    cb.type = "checkbox";\n'
        "    cb.checked = col.isVisible();\n"
        "    cb.dataset.field = col.getField();\n"
        '    cb.addEventListener("change", function() {\n'
        "      if (cb.checked) { col.show(); } else { col.hide(); }\n"
        "    });\n"
        "    label.appendChild(cb);\n"
        "    label.appendChild(document.createTextNode(col.getField()));\n"
        "    return label;\n"
        "  }\n"
        "\n"
        "  function refresh() {\n"
        '    list.textContent = "";\n'
        "    // Build a field -> column index for fast lookup.\n"
        "    const colByField = {};\n"
        "    __trfTable.getColumns().forEach(function(c) {\n"
        "      const f = c.getField();\n"
        "      if (f) { colByField[f] = c; }\n"
        "    });\n"
        "    const usedFields = new Set();\n"
        "    // Render each declared group, then a catchall 'Other' for any\n"
        "    // fields not present in any group.\n"
        "    __trfColumnGroups.forEach(function(group) {\n"
        '      const groupHeader = document.createElement("h4");\n'
        "      groupHeader.textContent = group.label;\n"
        "      list.appendChild(groupHeader);\n"
        "      // iter 8 — per-group Show all / Hide all buttons. Placed\n"
        "      // BETWEEN the group header (<h4>) and the optional footnote\n"
        "      // (<div class='trf-group-footnote'>) per user spec at iter-7\n"
        "      // feedback. Each button iterates only the columns declared\n"
        "      // for THIS group (group.columns) and calls show()/hide() on\n"
        "      // each, then refresh()es the sidebar so checkbox states\n"
        "      // re-hydrate. Distinct from the sidebar-header Show all /\n"
        "      // Hide all buttons (which toggle ALL columns globally) by\n"
        "      // CSS class .trf-group-toggle vs .trf-toggle-all.\n"
        '      const groupBtnRow = document.createElement("div");\n'
        '      groupBtnRow.className = "trf-group-toggle-row";\n'
        '      groupBtnRow.style.marginBottom = "3px";\n'
        '      const groupShowBtn = document.createElement("button");\n'
        '      groupShowBtn.type = "button";\n'
        '      groupShowBtn.className = "trf-group-toggle";\n'
        '      groupShowBtn.dataset.action = "show-all";\n'
        '      groupShowBtn.textContent = "Show all";\n'
        '      const groupHideBtn = document.createElement("button");\n'
        '      groupHideBtn.type = "button";\n'
        '      groupHideBtn.className = "trf-group-toggle";\n'
        '      groupHideBtn.dataset.action = "hide-all";\n'
        '      groupHideBtn.textContent = "Hide all";\n'
        "      // Closure-capture group.columns into the click handler — the\n"
        "      // forEach var rebinds across iterations so we must capture\n"
        "      // group.columns by value at handler-attach time.\n"
        "      (function(groupColumns) {\n"
        '        groupShowBtn.addEventListener("click", function() {\n'
        "          groupColumns.forEach(function(field) {\n"
        "            if (colByField[field]) { colByField[field].show(); }\n"
        "          });\n"
        "          refresh();\n"
        "        });\n"
        '        groupHideBtn.addEventListener("click", function() {\n'
        "          groupColumns.forEach(function(field) {\n"
        "            if (colByField[field]) { colByField[field].hide(); }\n"
        "          });\n"
        "          refresh();\n"
        "        });\n"
        "      })(group.columns);\n"
        "      groupBtnRow.appendChild(groupShowBtn);\n"
        "      groupBtnRow.appendChild(groupHideBtn);\n"
        "      list.appendChild(groupBtnRow);\n"
        "      if (group.footnote) {\n"
        '        const fn = document.createElement("div");\n'
        '        fn.className = "trf-group-footnote";\n'
        "        fn.textContent = group.footnote;\n"
        "        list.appendChild(fn);\n"
        "      }\n"
        "      group.columns.forEach(function(field) {\n"
        "        if (colByField[field]) {\n"
        "          list.appendChild(makeCheckboxRow(colByField[field]));\n"
        "          usedFields.add(field);\n"
        "        }\n"
        "      });\n"
        "    });\n"
        "    const otherCols = __trfTable.getColumns().filter(function(c) {\n"
        "      const f = c.getField();\n"
        "      return f && !usedFields.has(f);\n"
        "    });\n"
        "    if (otherCols.length > 0) {\n"
        '      const otherHeader = document.createElement("h4");\n'
        '      otherHeader.textContent = __trfColumnGroups.length > 0 ? "Other" : "All columns";\n'
        "      list.appendChild(otherHeader);\n"
        "      otherCols.forEach(function(col) {\n"
        "        list.appendChild(makeCheckboxRow(col));\n"
        "      });\n"
        "    }\n"
        "  }\n"
        "  refresh();\n"
        '  sidebar.querySelectorAll(".trf-toggle-all").forEach(function(btn) {\n'
        '    btn.addEventListener("click", function() {\n'
        '      const show = btn.dataset.action === "show-all";\n'
        "      __trfTable.getColumns().forEach(function(col) {\n"
        "        if (!col.getField()) { return; }\n"
        "        if (show) { col.show(); } else { col.hide(); }\n"
        "      });\n"
        "      refresh();\n"
        "    });\n"
        "  });\n"
        "  // iter 5b — Copy-table button. Builds TSV manually from the\n"
        "  // table's active (filtered) row set + currently-visible columns,\n"
        "  // then writes to clipboard via the navigator.clipboard API.\n"
        "  // Avoids __trfTable.setClipboardCopyConfig (does NOT exist in\n"
        "  // Tabulator v6.4 — the option is construction-time only; calling\n"
        "  // the non-existent setter at runtime threw the 'is not a function'\n"
        "  // error the user surfaced in screenshot at scratch L4255).\n"
        "  // Headers ARE included in the whole-table copy per user spec at\n"
        "  // scratch L4260; drag-select / Ctrl+C-row copies handled by\n"
        "  // Tabulator's native clipboard module without headers (config in\n"
        "  // build_options_dict).\n"
        '  const copyBtn = sidebar.querySelector(".trf-copy-table");\n'
        '  const copyStatus = sidebar.querySelector(".trf-copy-status");\n'
        "  if (copyBtn) {\n"
        '    copyBtn.addEventListener("click", async function() {\n'
        "      try {\n"
        "        const visibleCols = __trfTable.getColumns().filter(function(c) {\n"
        "          return c.getField() && c.isVisible();\n"
        "        });\n"
        "        const fields = visibleCols.map(function(c) { return c.getField(); });\n"
        "        const headerRow = fields.join('\\t');\n"
        '        const rows = __trfTable.getData("active");\n'
        "        const dataRows = rows.map(function(row) {\n"
        "          return fields.map(function(f) {\n"
        '            const v = row[f];\n'
        '            return (v === null || v === undefined) ? "" : String(v);\n'
        "          }).join('\\t');\n"
        "        });\n"
        '        const tsv = [headerRow].concat(dataRows).join("\\n");\n'
        "        if (navigator.clipboard && navigator.clipboard.writeText) {\n"
        "          await navigator.clipboard.writeText(tsv);\n"
        "        } else {\n"
        "          // Fallback for older browsers / non-secure contexts.\n"
        '          const ta = document.createElement("textarea");\n'
        "          ta.value = tsv;\n"
        '          ta.style.position = "fixed";\n'
        '          ta.style.left = "-9999px";\n'
        "          document.body.appendChild(ta);\n"
        "          ta.select();\n"
        '          document.execCommand("copy");\n'
        "          document.body.removeChild(ta);\n"
        "        }\n"
        '        copyStatus.textContent = '
        '"Copied " + dataRows.length + " rows × " + fields.length + " cols to clipboard.";\n'
        "        setTimeout(function() {\n"
        '          copyStatus.textContent = "";\n'
        "        }, 3000);\n"
        "      } catch (e) {\n"
        '        copyStatus.textContent = "Copy failed: " + (e.message || e);\n'
        "      }\n"
        "    });\n"
        "  }\n"
        "  // iter 9.2 — Reset all button handler. Clears every\n"
        "  // localStorage key matching the `tabulator-{persistenceID}-`\n"
        "  // prefix (Persistence.js:94 + defaults/readers.js:4 establish\n"
        "  // this key format), then reloads the page so Tabulator\n"
        "  // re-instantiates without any persisted state. This is the\n"
        "  // user-facing escape hatch for stale-localStorage bugs (e.g.,\n"
        "  // the iter-9 alignment toggle persisting after a config\n"
        "  // change that should have disabled column persistence). When\n"
        "  // persistenceID is empty (no persistence configured), the\n"
        "  // button is hidden because the clear has no target.\n"
        '  const resetBtn = sidebar.querySelector(".trf-reset-all");\n'
        '  const __trfPersistenceID = (tableOptions && tableOptions.persistenceID) || "";\n'
        "  if (resetBtn) {\n"
        "    if (!__trfPersistenceID) {\n"
        '      resetBtn.style.display = "none";\n'
        "    } else {\n"
        '      resetBtn.addEventListener("click", function() {\n'
        '        const prefix = "tabulator-" + __trfPersistenceID + "-";\n'
        "        const keysToRemove = [];\n"
        "        for (let i = 0; i < localStorage.length; i++) {\n"
        "          const k = localStorage.key(i);\n"
        "          if (k && k.indexOf(prefix) === 0) { keysToRemove.push(k); }\n"
        "        }\n"
        "        keysToRemove.forEach(function(k) { localStorage.removeItem(k); });\n"
        "        if (window.console && window.console.log) {\n"
        "          window.console.log('trf reset: removed ' + keysToRemove.length + "
        "' localStorage keys with prefix ' + prefix);\n"
        "        }\n"
        "        location.reload();\n"
        "      });\n"
        "    }\n"
        "  }\n"
        "  // Keep sidebar in sync when columns are toggled from elsewhere\n"
        "  // (e.g., persistence-restore on load).\n"
        '  __trfTable.on("columnVisibilityChanged", refresh);\n'
        "});\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for non-native types (numpy scalars, etc.)."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
