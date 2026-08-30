# Add TypeScript Types to JavaScript

**Category:** Code transformation
**Target:** Type inference, generics, strict mode, interface design

---

## Prompt

Add strict TypeScript types to the following JavaScript module. This is an event-driven state machine for a shopping cart.

**Source (JavaScript):**

```javascript
class CartStateMachine {
    constructor(config) {
        this.state = 'empty';
        this.items = [];
        this.listeners = {};
        this.config = config || {};
        this.history = [];
        this.metadata = {};
    }

    on(event, handler) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(handler);
        return () => {
            this.listeners[event] = this.listeners[event].filter(h => h !== handler);
        };
    }

    emit(event, data) {
        const handlers = this.listeners[event] || [];
        handlers.forEach(h => h(data));
    }

    addItem(item) {
        if (this.state === 'checked_out') {
            throw new Error('Cannot add items after checkout');
        }

        const existing = this.items.find(i => i.id === item.id);
        const oldState = this.state;

        if (existing) {
            existing.quantity += item.quantity || 1;
        } else {
            this.items.push({ ...item, quantity: item.quantity || 1 });
        }

        if (oldState === 'empty') {
            this.state = 'has_items';
        }

        this.history.push({
            action: 'add',
            item: item.id,
            timestamp: Date.now(),
            prevState: oldState,
            newState: this.state
        });

        this.emit('itemAdded', { item, state: this.state });
        return this;
    }

    removeItem(itemId) {
        const index = this.items.findIndex(i => i.id === itemId);
        if (index === -1) return this;

        const removed = this.items.splice(index, 1)[0];
        const oldState = this.state;

        if (this.items.length === 0) {
            this.state = 'empty';
        }

        this.history.push({
            action: 'remove',
            item: itemId,
            timestamp: Date.now(),
            prevState: oldState,
            newState: this.state
        });

        this.emit('itemRemoved', { item: removed, state: this.state });
        return this;
    }

    updateQuantity(itemId, quantity) {
        const item = this.items.find(i => i.id === itemId);
        if (!item) return this;

        if (quantity <= 0) {
            return this.removeItem(itemId);
        }

        item.quantity = quantity;
        this.emit('quantityUpdated', { item, state: this.state });
        return this;
    }

    applyDiscount(code, value) {
        if (!this.config.discounts) {
            this.config.discounts = {};
        }
        this.config.discounts[code] = value;
        this.emit('discountApplied', { code, value });
        return this;
    }

    calculateTotal() {
        let subtotal = this.items.reduce((sum, item) => {
            return sum + (item.price * item.quantity);
        }, 0);

        let discount = 0;
        if (this.config.discounts) {
            Object.values(this.config.discounts).forEach(d => {
                if (typeof d === 'number') discount += d;
            });
        }

        return {
            subtotal,
            discount,
            total: Math.max(0, subtotal - discount),
            itemCount: this.items.reduce((sum, i) => sum + i.quantity, 0)
        };
    }

    checkout(paymentInfo) {
        if (this.state === 'empty') {
            throw new Error('Cannot checkout empty cart');
        }

        const total = this.calculateTotal();
        const oldState = this.state;
        this.state = 'checked_out';

        this.history.push({
            action: 'checkout',
            timestamp: Date.now(),
            prevState: oldState,
            newState: this.state,
            paymentMethod: paymentInfo.method
        });

        this.emit('checkedOut', { total, paymentInfo });
        return { success: true, total, orderId: this.generateId() };
    }

    reset() {
        const oldState = this.state;
        this.state = 'empty';
        this.items = [];
        this.history = [];
        this.emit('reset', { prevState: oldState });
        return this;
    }

    getHistory() {
        return [...this.history];
    }

    generateId() {
        return Math.random().toString(36).substring(2, 10);
    }
}

module.exports = { CartStateMachine };
```

**Requirements:**

- Convert to `.ts` with `strict: true` compiler options
- Define interfaces for: `CartItem`, `CartState`, `HistoryEntry`, `CartConfig`, `DiscountMap`, `PaymentInfo`, `CheckoutResult`, `CartTotals`
- Use union types for state (`'empty' | 'has_items' | 'checked_out'`)
- Use generics where appropriate (event listener map)
- Add proper return types (use `this` for fluent methods)
- Use `readonly` where applicable
- Include a test file with `vitest` or `jest` that exercises all methods
- Include `tsconfig.json` with strict settings

**Constraints:**

- No `any` types allowed
- No `as` casts unless absolutely necessary (with justification comment)
- Must compile with `tsc --noEmit` with zero errors
- Tests must pass

Produce all files with complete working code. No placeholders, no TODOs.
