# Gradio Interface Redesign Prompt

## Design System Overview

Create a modern, minimal dark theme interface for a Gradio application with the following comprehensive design specifications:

---

## Color Palette

### Background Colors

- **Primary Background**: `#0a0a0b` (near-black for main app background)
- **Card Background (Subtle)**: `rgba(24, 24, 27, 0.3)` or Tailwind `zinc-900/30`
- **Card Background (Stronger)**: `rgba(24, 24, 27, 0.5)` or Tailwind `zinc-900/50`
- **Secondary Background**: `#18181b` (zinc-900 for solid backgrounds)

### Border Colors

- **Primary Border**: `#27272a` (zinc-800)
- **Hover Border**: `#3f3f46` (zinc-700)
- **Focus Border**: Transparent (replaced by ring)

### Text Colors

- **Primary Text**: `#ffffff` (white)
- **Secondary Text**: `#d4d4d8` (zinc-300)
- **Tertiary Text**: `#a1a1aa` (zinc-400)
- **Placeholder Text**: `#71717a` (zinc-500)
- **Disabled Text**: `#52525b` (zinc-600)

### Accent Colors

- **Primary Accent**: `#4f46e5` (indigo-600 for buttons)
- **Primary Accent Hover**: `#4338ca` (indigo-700)
- **Primary Accent Light**: `#6366f1` (indigo-500 for highlights)
- **Focus Ring**: `rgba(79, 70, 229, 0.5)` (indigo-500/50)

### Status Colors

- **Success/Active**: `#10b981` (emerald-500)
- **Warning Background**: `rgba(245, 158, 11, 0.05)` (amber-500/5)
- **Warning Border**: `rgba(245, 158, 11, 0.2)` (amber-500/20)
- **Warning Text**: `#f59e0b` (amber-500)
- **Info**: `#3b82f6` (blue-500)

---

## Spacing System

### Padding

- **Input Fields**: `16px 16px` (px-4 py-2.5)
- **Small Cards**: `16px` (p-4)
- **Large Cards**: `24px` (p-6)
- **Large Upload Areas**: `48px` (p-12)
- **Buttons (Standard)**: `16px 16px` (px-4 py-2.5)
- **Buttons (Primary)**: `24px 12px` (px-6 py-3)

### Margins

- **Small Gap**: `8px` (gap-2 or mb-2)
- **Medium Gap**: `12px` (gap-3 or mb-3)
- **Standard Gap**: `16px` (gap-4 or mb-4)
- **Large Gap**: `24px` (gap-6 or mb-6)
- **Section Gap**: `32px` (gap-8 or mb-8)

### Container Spacing

- **Container Max Width**: `1280px` (max-w-7xl)
- **Container Horizontal Padding**: `24px` (px-6)
- **Container Vertical Padding**: `32px` (py-8)

---

## Border Radius

- **Standard Radius**: `8px` (rounded-lg for all inputs, buttons, cards)
- **Small Radius**: `6px` (rounded-md for smaller elements)

---

## Border Styles

- **Standard Border**: `1px solid #27272a`
- **Dashed Border (Upload Zones)**: `2px dashed #27272a`
- **Focus Ring**: `2px solid rgba(79, 70, 229, 0.5)` with `0px` border on focus

---

## Typography

### Font Sizes

- **Headings (h1)**: Default system size (do not override with Tailwind classes)
- **Headings (h2)**: Default system size (do not override with Tailwind classes)
- **Body Text**: Default system size (do not override with Tailwind classes)
- **Small Text (Labels)**: `14px` (text-sm)
- **Extra Small Text (Helper)**: `12px` (text-xs)

### Font Weights

- Use default font weights from the design system
- Do not add font-bold or font-semibold classes unless specifically needed

---

## Component Specifications

### Input Fields

```
- Background: zinc-900/50
- Border: 1px solid zinc-800
- Border Radius: 8px
- Padding: 16px (px-4 py-2.5)
- Text Color: white
- Placeholder Color: zinc-600
- Focus State:
  - Outline: none
  - Ring: 2px indigo-500/50
  - Border: transparent
- Hover State: No change (focus only)
- Transition: transition-all
```

### Buttons (Primary)

```
- Background: indigo-600
- Text Color: white
- Padding: 12px 24px (px-6 py-3)
- Border Radius: 8px
- Hover Background: indigo-700
- Transition: transition-colors
- Icon Spacing: gap-2
```

### Buttons (Secondary)

```
- Background: zinc-900/50
- Border: 1px solid zinc-800
- Text Color: zinc-300
- Padding: 10px 16px (px-4 py-2.5)
- Border Radius: 8px
- Hover Border: zinc-700
- Transition: transition-all
```

### Cards

```
- Background: zinc-900/30
- Border: 1px solid zinc-800
- Border Radius: 8px
- Padding: 24px (p-6 for content cards)
- Padding: 16px (p-4 for info cards)
```

### Upload Zones (Drag & Drop)

```
- Border: 2px dashed zinc-800
- Border Radius: 8px
- Padding: 48px (p-12)
- Background (Default): transparent
- Background (Drag Active): indigo-500/5
- Border (Drag Active): indigo-500
- Icon Color: zinc-600
- Icon Size: 40px (w-10 h-10)
- Transition: transition-all
```

### Tabs

```
- Tab Container: border-bottom 1px solid zinc-800
- Tab Button Padding: 12px 16px (px-4 py-3)
- Tab Text (Inactive): zinc-500
- Tab Text (Active): white
- Tab Text (Hover): zinc-300
- Active Indicator: 2px height (h-0.5)
- Active Indicator Color: indigo-500
- Transition: transition-colors
```

### Select Dropdowns

```
- Same as Input Fields
- Icon (ChevronDown): zinc-500, 16px (w-4 h-4)
- Icon Position: absolute right-16px
- Appearance: none (hide default arrow)
```

### Chat Messages

```
User Message:
- Background: indigo-600
- Text Color: white
- Padding: 10px 16px (px-4 py-2.5)
- Border Radius: 8px
- Max Width: 80%
- Alignment: right

Bot Message:
- Background: zinc-800
- Text Color: zinc-100
- Padding: 10px 16px (px-4 py-2.5)
- Border Radius: 8px
- Max Width: 80%
- Alignment: left

Chat Container:
- Background: zinc-900/30
- Border: 1px solid zinc-800
- Border Radius: 8px
- Height: 400px
- Padding: 24px (p-6)
- Overflow: auto
```

### Info/Warning Boxes

```
- Background: amber-500/5
- Border: 1px solid amber-500/20
- Border Radius: 8px
- Padding: 16px (p-4)
- Icon Color: amber-500
- Icon Size: 20px (w-5 h-5)
- Text Color: zinc-400
- Heading Color: amber-500
```

### Status Indicators

```
- Active/Success Icon Color: emerald-500
- Warning Icon Color: amber-500
- Info Icon Color: blue-500
- Icon Size: 16px (w-4 h-4)
```

---

## Layout Structure

### Grid System

```
- Two Column Layout: grid grid-cols-1 lg:grid-cols-2 gap-6
- Three Column Layout: grid grid-cols-1 lg:grid-cols-3 gap-6
- Form Fields: grid grid-cols-1 md:grid-cols-3 gap-4
```

### Responsive Breakpoints

```
- Mobile: < 768px (default)
- Tablet: md: >= 768px
- Desktop: lg: >= 1024px
```

---

## Interactive States

### Focus States

```
- Remove default outline: outline-none
- Add ring: ring-2 ring-indigo-500/50
- Make border transparent: border-transparent
- Transition: transition-all
```

### Hover States

```
Buttons:
- Primary: background changes from indigo-600 to indigo-700
- Secondary: border changes from zinc-800 to zinc-700

Links/Text:
- Color changes from zinc-500 to zinc-300

Transition: transition-colors or transition-all
```

### Active/Selected States

```
- Background: indigo-600
- Text: white
- No border needed for selected buttons
```

---

## Icons

### Icon Library

- Use Lucide React icons
- Standard icon size: 16px (w-4 h-4)
- Large icon size: 20px (w-5 h-5)
- Upload area icon size: 40px (w-10 h-10)

### Common Icons Used

- Upload
- Send
- RefreshCw
- ChevronDown
- FileText
- Info
- Database
- BarChart3
- Link

---

## Additional Guidelines

### Shadows

- Avoid heavy shadows; use subtle borders instead
- Rely on background opacity and borders for depth

### Animations

- Use `transition-all` for multi-property changes
- Use `transition-colors` for color-only changes
- Keep transitions smooth and subtle

### Text Hierarchy

- Use color and size (not weight) for hierarchy
- Primary: white
- Secondary: zinc-300
- Tertiary: zinc-400
- Placeholder/Disabled: zinc-500, zinc-600

### Form Labels

- Position: above input field
- Text Size: text-sm (14px)
- Text Color: zinc-300
- Margin Bottom: mb-2 (8px)

### Helper Text

- Text Size: text-xs (12px)
- Text Color: zinc-500
- Margin Bottom: mb-3 (12px)

---

## Example Component Implementations

### Text Input

```css
Background: zinc-900/50 (#18181b with 50% opacity)
Border: 1px solid #27272a
Border Radius: 8px
Padding: 10px 16px
Color: #ffffff
Placeholder Color: #52525b
Focus Ring: 2px solid rgba(79, 70, 229, 0.5)
Focus Border: transparent
```

### Primary Button

```css
Background: #4f46e5
Color: #ffffff
Border Radius: 8px
Padding: 12px 24px
Hover Background: #4338ca
```

### Card Container

```css
Background: rgba(24, 24, 27, 0.3)
Border: 1px solid #27272a
Border Radius: 8px
Padding: 24px
```

---

## Implementation Notes

1. **Consistency**: All interactive elements use the same border radius (8px) and similar padding patterns
2. **Spacing**: Use multiples of 4px or 8px for all spacing
3. **No Emojis**: Keep the design professional without decorative emojis
4. **Accessibility**: Ensure sufficient contrast between text and backgrounds
5. **Focus Management**: All interactive elements must have visible focus states
6. **Responsive**: Components should work on mobile, tablet, and desktop
7. **Loading States**: Consider adding subtle loading indicators using zinc-600 color
8. **Empty States**: Use zinc-600 text color for empty state messages

---

## Gradio-Specific Adaptations

When applying this design to Gradio:

1. **gr.Textbox**: Apply input field styles
2. **gr.Button**: Apply primary or secondary button styles based on importance
3. **gr.Dropdown**: Apply select dropdown styles
4. **gr.File**: Apply upload zone styles with dashed border
5. **gr.Chatbot**: Apply chat message styles
6. **gr.Row/Column**: Use grid system with consistent gaps
7. **gr.Tab**: Apply tab styles with active indicator
8. **gr.Markdown**: Use appropriate text colors (zinc-300 for body, white for headings)

### Gradio CSS Custom Properties

```css
--body-background-fill: #0a0a0b;
--background-fill-primary: rgba(24, 24, 27, 0.5);
--background-fill-secondary: rgba(24, 24, 27, 0.3);
--border-color-primary: #27272a;
--text-color-primary: #ffffff;
--text-color-secondary: #a1a1aa;
--color-accent: #4f46e5;
--color-accent-soft: rgba(79, 70, 229, 0.5);
--radius-lg: 8px;
--spacing-sm: 8px;
--spacing-md: 16px;
--spacing-lg: 24px;
```

---

This design system ensures a cohesive, modern, and minimal dark theme that provides excellent user experience while maintaining visual consistency throughout the interface.