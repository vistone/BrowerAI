// Drag and Drop Template
// A flexible drag and drop implementation supporting sorting and moving between containers

interface DragDropOptions {
    dragHandle: string;
    dropZones: string[];
    sortable?: boolean;
    onDragStart?: (element: HTMLElement) => void;
    onDragMove?: (element: HTMLElement, x: number, y: number) => void;
    onDragEnd?: (element: HTMLElement, dropZone: HTMLElement | null) => void;
    onReorder?: (newOrder: HTMLElement[]) => void;
}

class DragDrop {
    private options: DragDropOptions;
    private draggedElement: HTMLElement | null = null;
    private placeholder: HTMLElement | null = null;
    private startX: number = 0;
    private startY: number = 0;
    private initialX: number = 0;
    private initialY: number = 0;
    private currentDropZone: HTMLElement | null = null;

    constructor(options: DragDropOptions) {
        this.options = {
            sortable: true,
            ...options
        };
        this.init();
    }

    private init() {
        const dragHandles = document.querySelectorAll(this.options.dragHandle);
        dragHandles.forEach(handle => {
            handle.addEventListener('mousedown', this.onDragStart.bind(this));
            handle.addEventListener('touchstart', this.onDragStart.bind(this), { passive: false });
        });

        document.addEventListener('mousemove', this.onDragMove.bind(this));
        document.addEventListener('touchmove', this.onDragMove.bind(this), { passive: false });

        document.addEventListener('mouseup', this.onDragEnd.bind(this));
        document.addEventListener('touchend', this.onDragEnd.bind(this));
    }

    private onDragStart(e: MouseEvent | TouchEvent) {
        e.preventDefault();
        
        const target = e.target as HTMLElement;
        this.draggedElement = target.closest('.draggable') as HTMLElement;
        
        if (!this.draggedElement) return;

        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        this.startX = clientX;
        this.startY = clientY;

        const rect = this.draggedElement.getBoundingClientRect();
        this.initialX = rect.left;
        this.initialY = rect.top;

        // Create placeholder
        this.placeholder = document.createElement('div');
        this.placeholder.className = 'drag-placeholder';
        this.placeholder.style.height = `${rect.height}px`;
        this.draggedElement.parentNode?.insertBefore(this.placeholder, this.draggedElement);

        // Set drag styles
        this.draggedElement.classList.add('dragging');
        this.draggedElement.style.position = 'fixed';
        this.draggedElement.style.left = `${rect.left}px`;
        this.draggedElement.style.top = `${rect.top}px`;
        this.draggedElement.style.width = `${rect.width}px`;
        this.draggedElement.style.zIndex = '1000';

        this.options.onDragStart?.(this.draggedElement);
    }

    private onDragMove(e: MouseEvent | TouchEvent) {
        if (!this.draggedElement) return;

        e.preventDefault();

        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        const deltaX = clientX - this.startX;
        const deltaY = clientY - this.startY;

        this.draggedElement.style.left = `${this.initialX + deltaX}px`;
        this.draggedElement.style.top = `${this.initialY + deltaY}px`;

        // Check drop zone
        this.checkDropZone(clientX, clientY);

        // Update placeholder position
        this.updatePlaceholder(clientX, clientY);

        this.options.onDragMove?.(this.draggedElement, clientX, clientY);
    }

    private onDragEnd(e: MouseEvent | TouchEvent) {
        if (!this.draggedElement) return;

        // Drop to target zone
        if (this.currentDropZone && this.placeholder) {
            this.currentDropZone.insertBefore(this.draggedElement, this.placeholder);
        } else if (this.placeholder) {
            this.placeholder.parentNode?.insertBefore(this.draggedElement, this.placeholder);
        }

        // Cleanup
        this.draggedElement.classList.remove('dragging');
        this.draggedElement.style.position = '';
        this.draggedElement.style.left = '';
        this.draggedElement.style.top = '';
        this.draggedElement.style.width = '';
        this.draggedElement.style.zIndex = '';

        this.placeholder?.remove();
        this.placeholder = null;

        this.options.onDragEnd?.(this.draggedElement, this.currentDropZone);

        // Trigger reorder callback
        if (this.options.sortable) {
            const container = this.draggedElement.parentElement;
            if (container) {
                const newOrder = Array.from(container.querySelectorAll('.draggable')) as HTMLElement[];
                this.options.onReorder?.(newOrder);
            }
        }

        this.draggedElement = null;
        this.currentDropZone = null;
    }

    private checkDropZone(x: number, y: number) {
        this.currentDropZone = null;

        for (const selector of this.options.dropZones) {
            const zones = document.querySelectorAll(selector);
            for (const zone of zones) {
                const rect = zone.getBoundingClientRect();
                if (x >= rect.left && x <= rect.right && 
                    y >= rect.top && y <= rect.bottom) {
                    this.currentDropZone = zone as HTMLElement;
                    zone.classList.add('drag-over');
                } else {
                    zone.classList.remove('drag-over');
                }
            }
        }
    }

    private updatePlaceholder(x: number, y: number) {
        if (!this.placeholder || !this.options.sortable) return;

        const container = this.placeholder.parentElement;
        if (!container) return;

        const siblings = Array.from(container.children);
        const placeholderIndex = siblings.indexOf(this.placeholder);

        for (let i = 0; i < siblings.length; i++) {
            if (i === placeholderIndex) continue;

            const sibling = siblings[i] as HTMLElement;
            const rect = sibling.getBoundingClientRect();
            const midY = rect.top + rect.height / 2;

            if (y < midY && i < placeholderIndex) {
                container.insertBefore(this.placeholder, sibling);
                break;
            } else if (y > midY && i > placeholderIndex) {
                container.insertBefore(this.placeholder, sibling.nextSibling);
                break;
            }
        }
    }

    destroy() {
        // Cleanup event listeners
    }
}

export { DragDrop, DragDropOptions };
