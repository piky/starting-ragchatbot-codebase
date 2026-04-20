# Frontend Changes: Dark/Light Theme Toggle

## Feature: Theme Toggle Button

Added a toggle button that allows users to switch between dark and light themes.

### Files Modified

#### 1. `frontend/style.css`
- Added light theme CSS variables under `[data-theme="light"]` selector with WCAG AA compliant colors
- Added smooth transition animations for theme changes (0.3s ease)
- Added theme toggle button styles:
  - Fixed position in top-right corner
  - Circular button with 44px diameter
  - Hover effects with scale and color changes
  - Focus ring for keyboard navigation
  - Rotation animation on icon when toggling
- Added responsive styles for mobile (40px button on screens <768px)
- Added light-theme specific overrides for:
  - Code blocks with subtle background
  - Source item styling
  - Welcome message shadows
  - Scrollbar colors

#### 2. `frontend/index.html`
- Added theme toggle button with:
  - Sun icon (visible in dark mode)
  - Moon icon (visible in light mode)
  - Accessible attributes: `aria-label` and `title`
  - Positioned at top-right of page

#### 3. `frontend/script.js`
- Added theme management functions:
  - `getPreferredTheme()`: Checks localStorage, falls back to system preference
  - `setTheme()`: Updates DOM and saves to localStorage
  - `updateThemeIcon()`: Swaps sun/moon icons based on theme
  - `toggleTheme()`: Switches between themes with rotation animation
- Added theme toggle button event listeners (click and keyboard)
- Theme preference persisted in localStorage with key `theme-preference`

### Design Details

**Toggle Button:**
- Position: Fixed top-right corner (1rem from edges)
- Size: 44px diameter (40px on mobile)
- Background: Uses `--surface` and `--border-color` variables
- Hover: Scale to 1.05, primary color border
- Focus: 3px focus ring using `--focus-ring`
- Active: Scale to 0.95

**Icons:**
- Sun icon: 8 rays around a circle (shown in dark mode)
- Moon icon: Crescent moon (shown in light mode)
- Size: 22px (20px on mobile)
- Animation: 360° rotation on toggle (0.4s duration)

**Theme Colors (WCAG AA Compliant):**
| Variable | Dark Theme | Light Theme |
|----------|------------|-------------|
| --background | #0f172a | #f1f5f9 |
| --surface | #1e293b | #ffffff |
| --surface-hover | #334155 | #f8fafc |
| --text-primary | #f1f5f9 | #0f172a (15.7:1 contrast) |
| --text-secondary | #94a3b8 | #475569 (7.1:1 contrast) |
| --border-color | #334155 | #cbd5e1 |
| --primary-color | #2563eb | #1d4ed8 |
| --primary-hover | #1d4ed8 | #1e40af |

### Accessibility
- Button is keyboard-navigable (Tab to focus)
- Enter and Space keys trigger toggle
- `aria-label` provides screen reader context
- Focus ring visible for keyboard users
- Light theme colors meet WCAG AA standards (4.5:1 minimum contrast)
- Text primary: 15.7:1 contrast ratio (AAA)
- Text secondary: 7.1:1 contrast ratio (AAA)

### Implementation Details

**CSS Custom Properties:**
- All theme colors defined using CSS variables in `:root` (dark) and `[data-theme="light"]` (light)
- Elements reference variables (e.g., `var(--background)`, `var(--text-primary)`) for automatic theme adaptation

**Data Attribute:**
- Theme controlled via `data-theme` attribute on `<html>` element
- Set via `document.documentElement.setAttribute('data-theme', theme)`
- Values: `"dark"` (default) or `"light"`

**Theme-Adaptive Elements:**
- Background and surface colors
- Text colors (primary and secondary)
- Border colors
- Code blocks and preformatted text
- Scrollbars
- Buttons and interactive elements
- Source items
- Welcome message
- Loading animations
- Collapsible section arrows

**Visual Hierarchy Maintained:**
- Same border radius (12px default, 24px for inputs)
- Consistent spacing and padding
- Same shadow depth relative to theme brightness
- Identical typography scale
- Matching interaction states (hover, focus, active)