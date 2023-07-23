
function isElementVisible(elm) {
    // Liberally inspired by https://stackoverflow.com/a/5354536
    const rect = elm.getBoundingClientRect();
    const viewHeight = Math.max(document.documentElement.clientHeight, window.innerHeight);
    return !(rect.bottom < 0 || rect.top >= viewHeight);
}

class ScrollIntoViewInteraction {
    DELAY_BEFORE_SCROLLING_BACK = 400

    constructor(scrollArea) {
        this.scrollArea = scrollArea
        this.savedTop = null
        this.timeout = null
    }

    scrollTo(child) {
        if (this.savedTop === null) {
            this.savedTop = this.scrollArea.scrollTop
        }
        if (this.timeout !== null) {
            clearTimeout(this.timeout)
            this.timeout = null
        }

        child.scrollIntoView({
            block: 'center',
            behavior: 'smooth',
        })
    }

    resetLater() {
        if (this.savedTop === null)
            return;
        if (this.timeout !== null)
            clearTimeout(this.timeout)

        this.timeout = setTimeout(() => {
            this.scrollArea.scrollTo({
                top: this.savedTop,
                behavior: 'smooth',
            })
            this.timeout = null
            this.savedTop = null
        }, this.DELAY_BEFORE_SCROLLING_BACK)
    }
}



document.body.addEventListener('htmx:load', (evt) => {
    const valueElements = document.getElementsByClassName('value')

    // Scroll to the bottom at startup
    {
        const elm = document.getElementById('stack-view')
        elm.scrollTo({ top: elm.scrollHeight })
    }

    // const scrollIntoView = new ScrollIntoViewInteraction(stack.scrollArea)

    function setHighlighted(valueId) {
        const valueElements = document.getElementsByClassName('script--value')
        if (typeof valueId === 'string') {
            for (const elm of valueElements) {
                if (elm.dataset.mcjsValue === valueId) {
                    elm.classList.add('highlighted')
                }
            }

        } else {
            for (const elm of valueElements) {
                elm.classList.remove('highlighted')
            }
        }
    }

    for (const element of document.getElementsByClassName('script--value')) {
        const valueId = element.dataset.mcjsValue;
        if (! /^[-\w\d]+$/.test(valueId)) {
            console.warn(`${element}: invalid valueId: ${valueId}`)
            continue
        }

        element.onmouseenter = (event) => { setHighlighted(valueId) }
        element.onmouseleave = () => { setHighlighted(null) }
    }


})


