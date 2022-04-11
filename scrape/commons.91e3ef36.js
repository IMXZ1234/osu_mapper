(window.webpackJsonp = window.webpackJsonp || []).push([[0], {
    "+bOe": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        })), s.d(t, "b", (function () {
            return i
        })), s.d(t, "c", (function () {
            return a
        }));
        var r = s("/DQ7");

        function n() {
            a(!1)
        }

        function i() {
            a(!0)
        }

        function a(e, t) {
            const s = document.querySelector(".js-blackout");
            s instanceof HTMLElement && (s.style.opacity = e && null != t ? String(t) : "", Object(r.c)(s, e))
        }
    }, "+kdZ": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return i
            }));
            var r = s("is6n");

            function n(e) {
                var t;
                if (!(e instanceof HTMLElement)) return;
                const s = null !== (t = e.closest(".js-forum-post")) && void 0 !== t ? t : e.closest("form");
                if (!(s instanceof HTMLElement)) return;
                const r = s.querySelector("[name=body]");
                return r instanceof HTMLTextAreaElement ? r : void 0
            }

            class i {
                constructor() {
                    Object.defineProperty(this, "handleClear", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            this.clearInput(n(t))
                        }
                    }), Object.defineProperty(this, "handlePageLoad", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            for (const e of document.querySelectorAll(".js-forum-post-input")) this.handleRestore(null, e)
                        }
                    }), Object.defineProperty(this, "handlePostSaved", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.clearInput(n(e.target))
                        }
                    }), Object.defineProperty(this, "handleRestore", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            const s = n(t);
                            if (null == s) return;
                            const i = this.getKeyFromInput(s);
                            if (null == i) return;
                            const a = localStorage.getItem(i);
                            if (null != a && (s.value = a), i.startsWith("forum-post-input:topic:")) {
                                const e = `forum-topic-reply--${Object(r.a)().pathname}--text`,
                                    t = localStorage.getItem(e);
                                null != t && (localStorage.removeItem(e), localStorage.setItem(i, t), s.value = t)
                            }
                        }
                    }), Object.defineProperty(this, "onInput", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.currentTarget;
                            if (!(t instanceof HTMLTextAreaElement)) return;
                            const s = this.getKeyFromInput(t);
                            null != s && ("" === t.value ? localStorage.removeItem(s) : localStorage.setItem(s, t.value))
                        }
                    }), e(document).on("input change", ".js-forum-post-input", this.onInput).on("turbolinks:load", this.handlePageLoad).on("ajax:success", ".js-forum-post-input--form", this.handlePostSaved), e.subscribe("forum-post-input:restore", this.handleRestore), e.subscribe("forum-post-input:clear", this.handleClear)
                }

                clearInput(e) {
                    if (null == e) return;
                    const t = this.getKeyFromInput(e);
                    null != t && localStorage.removeItem(t)
                }

                getKeyFromInput(e) {
                    return this.prefixKey(e.dataset.forumPostInputId)
                }

                prefixKey(e) {
                    if (null != e && "" !== e) return `forum-post-input:${e}`
                }
            }
        }).call(this, s("5wds"))
    }, "/DQ7": function (e, t, s) {
        "use strict";

        function r(e) {
            null == e || e.setAttribute("data-visibility", "visible")
        }

        function n(e) {
            null == e || e.setAttribute("data-visibility", "hidden")
        }

        function i(e, t) {
            if (null == e) return;
            ((null != t ? t : !function (e) {
                return "hidden" !== (null == e ? void 0 : e.getAttribute("data-visibility"))
            }(e)) ? r : n)(e)
        }

        s.d(t, "a", (function () {
            return r
        })), s.d(t, "b", (function () {
            return n
        })), s.d(t, "c", (function () {
            return i
        }))
    }, "/HbY": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        function i(e) {
            return r.createElement("div", {className: Object(n.a)("la-ball-clip-rotate", e.modifiers)})
        }
    }, "/cS6": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return d
            }));
            var r = s("sHNI"), n = s("MtWa"), i = s("Hs9Z"), a = s("/G9H"), o = s("tSlR"), l = s("dTpI"), c = s("cX0L"),
                u = s("ss8h");

            class d extends a.Component {
                constructor(e) {
                    super(e), Object.defineProperty(this, "bn", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "beatmap-discussion-editor-insertion-menu"
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `editor-insertion-menu-${Object(c.a)()}`
                    }), Object.defineProperty(this, "hideInsertMenuTimer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "hoveredBlock", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "insertPosition", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "insertRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: a.createRef()
                    }), Object.defineProperty(this, "mouseOver", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "scrollContainer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledContainerMouseExit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledContainerMouseMove", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledMenuMouseEnter", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledMenuMouseExit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledScroll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "containerMouseMove", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (!e.originalEvent) return;
                            const t = e.originalEvent.clientY, s = this.scrollContainer.children[0].children;
                            let r = 0;
                            for (const i of s) {
                                if (t < i.getBoundingClientRect().top) {
                                    if (r > 0) {
                                        const e = s[r - 1];
                                        t < e.getBoundingClientRect().top + e.getBoundingClientRect().height / 2 && r--
                                    }
                                    break
                                }
                                r < s.length - 1 && r++
                            }
                            this.hoveredBlock = s[r];
                            const n = this.hoveredBlock.getBoundingClientRect();
                            t > n.top + n.height / 2 ? this.insertPosition = "below" : this.insertPosition = "above", this.updatePosition(), this.showMenu(), this.startHideTimer()
                        }
                    }), Object.defineProperty(this, "forceHideMenu", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.mouseOver = !1, this.hideMenu()
                        }
                    }), Object.defineProperty(this, "hideMenu", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.insertRef.current && !this.mouseOver && (this.insertRef.current.style.display = "none")
                        }
                    }), Object.defineProperty(this, "insertBlock", {
                        enumerable: !0, configurable: !0, writable: !0, value: e => {
                            var t, s, r, n, i, a;
                            const c = this.context,
                                u = null === (t = this.hoveredBlock) || void 0 === t ? void 0 : t.lastChild,
                                d = e.currentTarget.dataset.discussionType,
                                p = null === (s = this.props.currentBeatmap) || void 0 === s ? void 0 : s.id;
                            let m;
                            switch (d) {
                                case"suggestion":
                                case"problem":
                                case"praise":
                                    m = {beatmapId: p, children: [{text: ""}], discussionType: d, type: "embed"};
                                    break;
                                case"paragraph":
                                    m = {children: [{text: ""}], type: "paragraph"}
                            }
                            if (!m || !u) return;
                            let h, b = l.b.toSlateNode(c, u);
                            if (o.g.isText(b) && "" === b.text || o.b.isElement(b) && o.a.isEmpty(c, b)) if ("above" === this.insertPosition) {
                                const e = null === (n = null === (r = this.hoveredBlock) || void 0 === r ? void 0 : r.previousSibling) || void 0 === n ? void 0 : n.lastChild;
                                null != e ? (b = l.b.toSlateNode(c, e), h = o.a.end(c, l.b.findPath(c, b))) : h = {
                                    offset: 0,
                                    path: []
                                }
                            } else {
                                const e = null === (a = null === (i = this.hoveredBlock) || void 0 === i ? void 0 : i.previousSibling) || void 0 === a ? void 0 : a.lastChild;
                                null != e ? (b = l.b.toSlateNode(c, e), h = o.a.start(c, l.b.findPath(c, b))) : h = o.a.end(c, [])
                            } else {
                                const e = l.b.findPath(c, b);
                                h = "above" === this.insertPosition ? o.a.start(c, e) : o.a.end(c, e)
                            }
                            o.h.insertNodes(c, m, {at: h})
                        }
                    }), Object.defineProperty(this, "insertButton", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            let t = "fas fa-question";
                            switch (e) {
                                case"praise":
                                case"problem":
                                case"suggestion":
                                    t = r.a[e];
                                    break;
                                case"paragraph":
                                    t = "fas fa-indent"
                            }
                            return a.createElement("button", {
                                className: `${this.bn}__menu-button ${this.bn}__menu-button--${e}`,
                                "data-discussion-type": e,
                                onClick: this.insertBlock,
                                title: osu.trans(`beatmaps.discussions.review.insert-block.${e}`),
                                type: "button"
                            }, a.createElement("i", {className: t}))
                        }
                    }), Object.defineProperty(this, "menuMouseEnter", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.mouseOver = !0
                        }
                    }), Object.defineProperty(this, "menuMouseLeave", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.mouseOver = !1, this.startHideTimer()
                        }
                    }), this.throttledContainerMouseExit = Object(i.throttle)(() => {
                        setTimeout(this.hideMenu, 100)
                    }, 10), this.throttledContainerMouseMove = Object(i.throttle)(this.containerMouseMove, 10), this.throttledMenuMouseEnter = Object(i.throttle)(this.menuMouseEnter, 10), this.throttledMenuMouseExit = Object(i.throttle)(this.menuMouseLeave, 10), this.throttledScroll = Object(i.throttle)(this.forceHideMenu, 10)
                }

                componentDidMount() {
                    this.insertRef.current && (e(this.insertRef.current).on(`mouseenter.${this.eventId}`, this.throttledMenuMouseEnter), e(this.insertRef.current).on(`mouseleave.${this.eventId}`, this.throttledMenuMouseExit)), e(window).on(`scroll.${this.eventId}`, this.throttledScroll)
                }

                componentDidUpdate() {
                    this.forceHideMenu()
                }

                componentWillUnmount() {
                    e(window).off(`.${this.eventId}`), this.scrollContainer && e(this.scrollContainer).off(`.${this.eventId}`), this.insertRef.current && e(this.insertRef.current).off(`.${this.eventId}`)
                }

                render() {
                    return a.createElement(n.a, null, a.createElement("div", {
                        ref: this.insertRef,
                        className: `${this.bn}`
                    }, a.createElement("div", {className: `${this.bn}__body`}, a.createElement("i", {className: "fas fa-plus"}), a.createElement("div", {className: `${this.bn}__buttons`}, this.insertButton("suggestion"), this.insertButton("problem"), this.insertButton("praise"), this.insertButton("paragraph")))))
                }

                setScrollContainer(t) {
                    this.scrollContainer && e(this.scrollContainer).off(`.${this.eventId}`), this.scrollContainer = t, e(this.scrollContainer).on(`mousemove.${this.eventId}`, this.throttledContainerMouseMove), e(this.scrollContainer).on(`mouseleave.${this.eventId}`, this.throttledContainerMouseExit), e(this.scrollContainer).on(`scroll.${this.eventId}`, this.throttledScroll)
                }

                showMenu() {
                    this.insertRef.current && (this.insertRef.current.style.display = "flex")
                }

                startHideTimer() {
                    this.hideInsertMenuTimer && window.clearTimeout(this.hideInsertMenuTimer), this.hideInsertMenuTimer = window.setTimeout(this.hideMenu, 2e3)
                }

                updatePosition() {
                    if (!this.scrollContainer || !this.hoveredBlock || !this.insertRef.current) return;
                    const e = this.hoveredBlock.getBoundingClientRect(),
                        t = this.scrollContainer.getBoundingClientRect();
                    this.insertRef.current.style.left = `${t.left + 15}px`, this.insertRef.current.style.width = `${t.width - 30}px`, "above" === this.insertPosition ? this.insertRef.current.style.top = `${e.top - 10}px` : "below" === this.insertPosition && (this.insertRef.current.style.top = `${e.top + e.height - 10}px`)
                }
            }

            Object.defineProperty(d, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: u.a})
        }).call(this, s("5wds"))
    }, "/jJF": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        })), s.d(t, "c", (function () {
            return o
        })), s.d(t, "d", (function () {
            return l
        }));
        var r = s("f4vq"), n = s("phBA");
        const i = (e, t, s) => {
            "abort" !== t && (r.a.userLogin.showOnError(e, s) || r.a.userVerification.showOnError(e, s) || osu.popup(osu.xhrErrorMessage(e), "danger"))
        }, a = (e, t) => {
            i(t.jqXHR, t.textStatus, () => {
                var e;
                return null === (e = t.submit) || void 0 === e ? void 0 : e.call(t)
            })
        }, o = e => (t, s) => {
            i(t, s, e)
        }, l = e => o(Object(n.c)(e))
    }, "/p7V": function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return O
            }));
            var n, i, a, o = s("sHNI"), l = s("c1EF"), c = s("0h6b"), u = s("/G9H"), d = s("xZyo"), p = s("I8Ok"),
                m = s("iH//"), h = s("tX/w"), b = s("Ma8u"), f = s("uuPA"), v = s("y1is"), g = s("9CpU"), y = s("C3HX"),
                w = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, _ = {}.hasOwnProperty;
            a = u.createElement, i = "beatmap-discussion", n = function (e) {
                var t, s, r;
                return s = e.type, t = e.discussion, r = e.users, Object(p.div)({className: "user-list-popup user-list-popup--blank"}, t.votes[s] < 1 ? osu.trans("beatmaps.discussions.votes.none." + s) : a(u.Fragment, null, Object(p.div)({className: "user-list-popup__title"}, osu.trans("beatmaps.discussions.votes.latest." + s), ":"), t.votes.voters[s].map((function (e) {
                    var t;
                    return Object(p.a)({
                        href: Object(c.a)("users.show", {user: e}),
                        className: "js-usercard user-list-popup__user",
                        key: e,
                        "data-user-id": e
                    }, a(l.a, {user: null != (t = r[e]) ? t : [], modifiers: ["full"]}))
                })), t.votes[s] > t.votes.voters[s].length ? Object(p.div)({className: "user-list-popup__remainder-count"}, osu.transChoice("common.count.plus_others", t.votes[s] - t.votes.voters[s].length)) : void 0))
            };
            var O = function (t) {
                function s(e) {
                    this.toggleCollapse = w(this.toggleCollapse, this), this.timestamp = w(this.timestamp, this), this.resolvedSystemPostId = w(this.resolvedSystemPostId, this), this.post = w(this.post, this), this.canBeRepliedTo = w(this.canBeRepliedTo, this), this.canDownvote = w(this.canDownvote, this), this.isVisible = w(this.isVisible, this), this.isRead = w(this.isRead, this), this.isOwner = w(this.isOwner, this), this.emitSetHighlight = w(this.emitSetHighlight, this), this.doVote = w(this.doVote, this), this.showVoters = w(this.showVoters, this), this.refreshTooltip = w(this.refreshTooltip, this), this.getTooltipContent = w(this.getTooltipContent, this), this.displayVote = w(this.displayVote, this), this.postFooter = w(this.postFooter, this), this.postButtons = w(this.postButtons, this), this.render = w(this.render, this), this.componentDidUpdate = w(this.componentDidUpdate, this), this.componentWillUnmount = w(this.componentWillUnmount, this), s.__super__.constructor.call(this, e), this.eventId = "beatmap-discussion-entry-" + this.props.discussion.id, this.tooltips = {}
                }

                return function (e, t) {
                    for (var s in t) _.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.componentWillUnmount = function () {
                    var e;
                    return null != (e = this.voteXhr) ? e.abort() : void 0
                }, s.prototype.componentDidUpdate = function () {
                    var e, t, s, r;
                    for (r in t = [], e = this.tooltips) _.call(e, r) && (s = e[r], t.push(this.refreshTooltip(s.qtip("api"), r)));
                    return t
                }, s.prototype.render = function () {
                    var e, t, s, r, n, o;
                    return this.isVisible(this.props.discussion) && (this.props.discussion.starting_post || this.props.discussion.posts && 0 !== this.props.discussion.posts.length) ? (s = Object(h.a)(i + "__line", {resolved: this.props.discussion.resolved}), !1, this._resolvedSystemPostId = null, e = this.props.discussion.starting_post || this.props.discussion.posts[0], n = Object(h.a)(i, {
                        "horizontal-desktop": "review" !== this.props.discussion.message_type,
                        deleted: null != this.props.discussion.deleted_at,
                        highlighted: this.props.highlighted,
                        preview: this.props.preview,
                        review: "review" === this.props.discussion.message_type,
                        timeline: null != this.props.discussion.timestamp,
                        unread: !this.isRead(e)
                    }), n += " js-beatmap-discussion-jump", o = null != (r = this.props.users[this.props.discussion.user_id]) ? r : this.props.users.null, t = Object(m.a)({
                        beatmapset: this.props.beatmapset,
                        currentBeatmap: this.props.currentBeatmap,
                        discussion: this.props.discussion,
                        user: o
                    }), Object(p.div)({
                        className: n,
                        "data-id": this.props.discussion.id,
                        onClick: this.emitSetHighlight
                    }, Object(p.div)({className: i + "__timestamp hidden-xs"}, this.timestamp()), Object(p.div)({className: i + "__discussion"}, Object(p.div)({
                        className: i + "__top",
                        style: osu.groupColour(t)
                    }, Object(p.div)({className: i + "__top-user"}, a(y.a, {
                        user: o,
                        group: t,
                        hideStripe: !0
                    })), Object(p.div)({className: i + "__top-message"}, this.post(e, "discussion")), Object(p.div)({className: i + "__top-actions"}, this.props.preview ? void 0 : this.postButtons())), this.props.preview ? void 0 : this.postFooter(), Object(p.div)({className: s})))) : null
                }, s.prototype.postButtons = function () {
                    return Object(p.div)({className: i + "__actions"}, null != this.props.parentDiscussion ? Object(p.a)({
                        href: BeatmapDiscussionHelper.url({discussion: this.props.parentDiscussion}),
                        title: osu.trans("beatmap_discussions.review.go_to_parent"),
                        className: i + "__link-to-parent js-beatmap-discussion--jump"
                    }, Object(p.i)({className: "fas fa-tasks"})) : void 0, ["up", "down"].map((e = this, function (t) {
                        return Object(p.div)({
                            key: t,
                            "data-type": t,
                            className: i + "__action",
                            onMouseOver: e.showVoters,
                            onTouchStart: e.showVoters
                        }, e.displayVote(t))
                    })), Object(p.button)({
                        className: i + "__action " + i + "__action--with-line",
                        onClick: this.toggleCollapse
                    }, Object(p.div)({className: "beatmap-discussion-expand " + (this.props.collapsed ? void 0 : "beatmap-discussion-expand--expanded")}, Object(p.i)({className: "fas fa-chevron-down"}))));
                    var e
                }, s.prototype.postFooter = function () {
                    var e, t, s;
                    return Object(p.div)({className: i + "__expanded " + (this.props.collapsed ? "hidden" : void 0)}, Object(p.div)({className: i + "__replies"}, function () {
                        var r, n, i, a;
                        for (a = [], r = 0, n = (i = this.props.discussion.posts.slice(1)).length; r < n; r++) if (s = i[r], this.isVisible(s)) {
                            if (s.system && "resolved" === s.message.type) {
                                if (e = s.message.value, t === e) continue;
                                t = e
                            }
                            a.push(this.post(s, "reply"))
                        }
                        return a
                    }.call(this)), this.canBeRepliedTo() ? a(f.a, {
                        currentUser: this.props.currentUser,
                        beatmapset: this.props.beatmapset,
                        currentBeatmap: this.props.currentBeatmap,
                        discussion: this.props.discussion
                    }) : void 0)
                }, s.prototype.displayVote = function (e) {
                    var t, s, r, n, i, a, o, l, c, u, d;
                    if (d = "beatmap-discussion-vote", t = (i = function () {
                        switch (e) {
                            case"up":
                                return [1, "thumbs-up"];
                            case"down":
                                return [-1, "thumbs-down"]
                        }
                    }())[0], n = i[1], null != t) return s = null != (a = this.props.discussion.current_user_attributes) ? a.vote_score : void 0, c = d + " " + d + "--" + e, 0 !== (l = s === t ? 0 : t) && (c += " " + d + "--inactive"), u = null != (o = this.props.users[this.props.discussion.user_id]) ? o : this.props.users.null, r = this.isOwner() || u.is_bot || "down" === e && !this.canDownvote() || !this.canBeRepliedTo(), Object(p.button)({
                        className: c,
                        "data-score": l,
                        disabled: r,
                        onClick: this.doVote
                    }, Object(p.i)({className: "fas fa-" + n}), Object(p.span)({className: d + "__count"}, this.props.discussion.votes[e]))
                }, s.prototype.getTooltipContent = function (e) {
                    return Object(d.renderToStaticMarkup)(a(n, {
                        type: e,
                        discussion: this.props.discussion,
                        users: this.props.users
                    }))
                }, s.prototype.refreshTooltip = function (e, t) {
                    return null != e ? e.set("content.text", this.getTooltipContent(t)) : void 0
                }, s.prototype.showVoters = function (t) {
                    var s, r;
                    if (!(s = t.currentTarget)._tooltip) return s._tooltip = !0, r = s.getAttribute("data-type"), this.tooltips[r] = e(s).qtip({
                        style: {
                            classes: "user-list-popup",
                            def: !1,
                            tip: !1
                        },
                        content: {text: this.getTooltipContent(r)},
                        position: {at: "top center", my: "bottom center", viewport: e(window)},
                        show: {
                            delay: 100, ready: !0, solo: !0, effect: function () {
                                return e(this).fadeTo(110, 1)
                            }
                        },
                        hide: {
                            fixed: !0, delay: 500, effect: function () {
                                return e(this).fadeTo(250, 0)
                            }
                        }
                    })
                }, s.prototype.doVote = function (t) {
                    var s;
                    return Object(b.b)(), null != (s = this.voteXhr) && s.abort(), this.voteXhr = e.ajax(Object(c.a)("beatmapsets.discussions.vote", {discussion: this.props.discussion.id}), {
                        method: "PUT",
                        data: {beatmap_discussion_vote: {score: t.currentTarget.dataset.score}}
                    }).done((function (t) {
                        return e.publish("beatmapsetDiscussions:update", {beatmapset: t})
                    })).fail(osu.ajaxError).always(b.a)
                }, s.prototype.emitSetHighlight = function (t) {
                    if (!t.defaultPrevented) return e.publish("beatmapset-discussions:highlight", {discussionId: this.props.discussion.id})
                }, s.prototype.isOwner = function (e) {
                    return null == e && (e = this.props.discussion), null != this.props.currentUser.id && e.user_id === this.props.currentUser.id
                }, s.prototype.isRead = function (e) {
                    var t;
                    return (null != (t = this.props.readPostIds) ? t.has(e.id) : void 0) || this.isOwner(e) || this.props.preview
                }, s.prototype.isVisible = function (e) {
                    return null != e && (this.props.showDeleted || null == e.deleted_at)
                }, s.prototype.canDownvote = function () {
                    return this.props.currentUser.is_admin || this.props.currentUser.is_moderator || this.props.currentUser.is_bng
                }, s.prototype.canBeRepliedTo = function () {
                    return (!this.props.beatmapset.discussion_locked || BeatmapDiscussionHelper.canModeratePosts(this.props.currentUser)) && (null == this.props.discussion.beatmap_id || null == this.props.currentBeatmap.deleted_at)
                }, s.prototype.post = function (e, t) {
                    var s, r, n, i, o, l, c;
                    if (null != e.id) return i = e.system ? g.a : v.a, n = BeatmapDiscussionHelper.canModeratePosts(this.props.currentUser), r = this.isOwner(e) && e.id > this.resolvedSystemPostId() && !this.props.beatmapset.discussion_locked, s = "discussion" === t ? null != (o = this.props.discussion.current_user_attributes) ? o.can_destroy : void 0 : n || r, a(i, {
                        key: e.id,
                        beatmapset: this.props.beatmapset,
                        beatmap: this.props.currentBeatmap,
                        discussion: this.props.discussion,
                        post: e,
                        type: t,
                        read: this.isRead(e),
                        users: this.props.users,
                        user: null != (l = this.props.users[e.user_id]) ? l : this.props.users.null,
                        lastEditor: null != e.last_editor_id ? null != (c = this.props.users[e.last_editor_id]) ? c : this.props.users.null : void 0,
                        canBeEdited: this.props.currentUser.is_admin || r,
                        canBeDeleted: s,
                        canBeRestored: n,
                        currentUser: this.props.currentUser
                    })
                }, s.prototype.resolvedSystemPostId = function () {
                    var e, t;
                    return null == this._resolvedSystemPostId && (t = r.findLast(this.props.discussion.posts, (function (e) {
                        return null != e && e.system && "resolved" === e.message.type
                    })), this._resolvedSystemPostId = null != (e = null != t ? t.id : void 0) ? e : -1), this._resolvedSystemPostId
                }, s.prototype.timestamp = function () {
                    var e;
                    return e = "beatmap-discussion-timestamp", Object(p.div)({className: e}, null != this.props.discussion.timestamp && this.props.isTimelineVisible ? Object(p.div)({className: e + "__point"}) : void 0, Object(p.div)({className: e + "__icons-container"}, Object(p.div)({className: e + "__icons"}, Object(p.div)({className: e + "__icon"}, Object(p.span)({className: "beatmap-discussion-message-type beatmap-discussion-message-type--" + r.kebabCase(this.props.discussion.message_type)}, Object(p.i)({
                        className: o.a[this.props.discussion.message_type],
                        title: osu.trans("beatmaps.discussions.message_type." + this.props.discussion.message_type)
                    }))), this.props.discussion.resolved ? Object(p.div)({className: e + "__icon " + e + "__icon--resolved"}, Object(p.i)({
                        className: "far fa-check-circle",
                        title: osu.trans("beatmaps.discussions.resolved")
                    })) : void 0), Object(p.div)({className: e + "__text"}, BeatmapDiscussionHelper.formatTimestamp(this.props.discussion.timestamp))))
                }, s.prototype.toggleCollapse = function () {
                    return e.publish("beatmapset-discussions:collapse", {discussionId: this.props.discussion.id})
                }, s
            }(u.PureComponent)
        }).call(this, s("5wds"), s("Hs9Z"))
    }, "0VTr": function (e, t, s) {
        "use strict";
        (function (e) {
            var r = s("QUfv"), n = s("G27q"), i = s("WLnA"), a = s("0h6b"), o = s("Hs9Z"), l = s("lv9K"), c = s("f4vq"),
                u = s("tz7b"), d = s("8gxX"), p = function (e, t, s, r) {
                    var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };
            let m = class {
                constructor() {
                    Object.defineProperty(this, "connectionStatus", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "disconnected"
                    }), Object.defineProperty(this, "hasConnectedOnce", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "userId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: null
                    }), Object.defineProperty(this, "active", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "endpoint", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "retryDelay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new d.a
                    }), Object.defineProperty(this, "timeout", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "ws", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "xhrLoadingState", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "handleNewEvent", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = this.parseMessageEvent(t);
                            null != s && ((e => "logout" === e.event)(s) ? (this.destroy(), c.a.userLoginObserver.logout()) : (e => "verified" === e.event)(s) ? e.publish("user-verification:success") : Object(i.a)(new u.a(s)))
                        }
                    }), Object.defineProperty(this, "reconnectWebSocket", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.connectionStatus = "disconnected", this.active && (this.timeout.connectWebSocket = window.setTimeout(Object(l.f)(() => {
                                this.ws = null, this.connectWebSocket()
                            }), this.retryDelay.get()))
                        }
                    }), Object.defineProperty(this, "startWebSocket", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (null != this.endpoint) return this.connectWebSocket();
                            this.xhrLoadingState.startWebSocket || (window.clearTimeout(this.timeout.startWebSocket), this.xhrLoadingState.startWebSocket = !0, this.xhr.startWebSocket = e.get(Object(a.a)("notifications.endpoint")).always(Object(l.f)(() => {
                                this.xhrLoadingState.startWebSocket = !1
                            })).done(Object(l.f)(e => {
                                this.retryDelay.reset(), this.endpoint = e.url, this.connectWebSocket()
                            })).fail(Object(l.f)(e => {
                                401 !== e.status ? this.timeout.startWebSocket = window.setTimeout(this.startWebSocket, this.retryDelay.get()) : this.destroy()
                            })))
                        }
                    }), Object(l.p)(this), Object(l.t)(() => this.isConnected, e => Object(i.a)(new n.a(e)), {fireImmediately: !0})
                }

                get isConnected() {
                    return "connected" === this.connectionStatus
                }

                boot() {
                    this.active = null != this.userId, this.active && this.startWebSocket()
                }

                handleDispatchAction(e) {
                    var t, s;
                    (null === (t = this.ws) || void 0 === t ? void 0 : t.readyState) === WebSocket.OPEN && e instanceof r.a && (null === (s = this.ws) || void 0 === s || s.send(JSON.stringify(e.message)))
                }

                setUserId(e) {
                    e !== this.userId && (this.active && this.destroy(), this.userId = e, this.boot())
                }

                connectWebSocket() {
                    if (!this.active || null == this.endpoint || null != this.ws) return;
                    this.connectionStatus = "connecting", window.clearTimeout(this.timeout.connectWebSocket);
                    const e = document.querySelector("meta[name=csrf-token]");
                    if (null == e) return;
                    const t = e.getAttribute("content");
                    this.ws = new WebSocket(`${this.endpoint}?csrf=${t}`), this.ws.addEventListener("open", Object(l.f)(() => {
                        this.retryDelay.reset(), this.connectionStatus = "connected", this.hasConnectedOnce = !0
                    })), this.ws.addEventListener("close", this.reconnectWebSocket), this.ws.addEventListener("message", this.handleNewEvent)
                }

                destroy() {
                    this.connectionStatus = "disconnecting", this.userId = null, this.active = !1, Object(o.forEach)(this.xhr, e => null == e ? void 0 : e.abort()), Object(o.forEach)(this.timeout, e => window.clearTimeout(e)), null != this.ws && (this.ws.close(), this.ws = null), this.connectionStatus = "disconnected"
                }

                parseMessageEvent(e) {
                    try {
                        const t = JSON.parse(e.data);
                        if (Object(u.b)(t)) return t;
                        console.debug("message missing event type.")
                    } catch (t) {
                        console.debug("Failed parsing data:", e.data)
                    }
                }
            };
            p([l.q], m.prototype, "connectionStatus", void 0), p([l.q], m.prototype, "hasConnectedOnce", void 0), p([l.q], m.prototype, "active", void 0), p([l.h], m.prototype, "isConnected", null), p([l.f], m.prototype, "connectWebSocket", null), p([l.f], m.prototype, "destroy", null), p([l.f], m.prototype, "reconnectWebSocket", void 0), m = p([i.b], m), t.a = m
        }).call(this, s("5wds"))
    }, "0h6b": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return D
        }));
        var r = s("is6n");
        const n = {
            url: "https://osu.ppy.sh", port: null, defaults: {}, routes: {
                "admin.beatmapsets.covers": {uri: "admin/beatmapsets/{beatmapset}/covers", methods: ["GET", "HEAD"]},
                "admin.beatmapsets.covers.regenerate": {
                    uri: "admin/beatmapsets/{beatmapset}/covers/regenerate",
                    methods: ["POST"]
                },
                "admin.beatmapsets.covers.remove": {
                    uri: "admin/beatmapsets/{beatmapset}/covers/remove",
                    methods: ["POST"]
                },
                "admin.beatmapsets.show": {uri: "admin/beatmapsets/{beatmapset}", methods: ["GET", "HEAD"]},
                "admin.beatmapsets.update": {uri: "admin/beatmapsets/{beatmapset}", methods: ["PUT", "PATCH"]},
                "admin.contests.get-zip": {uri: "admin/contests/{contest}/zip", methods: ["POST"]},
                "admin.contests.index": {uri: "admin/contests", methods: ["GET", "HEAD"]},
                "admin.contests.show": {uri: "admin/contests/{contest}", methods: ["GET", "HEAD"]},
                "admin.user-contest-entries.destroy": {
                    uri: "admin/user-contest-entries/{user_contest_entry}",
                    methods: ["DELETE"]
                },
                "admin.user-contest-entries.restore": {
                    uri: "admin/user-contest-entries/{user_contest_entry}/restore",
                    methods: ["POST"]
                },
                "admin.logs.index": {uri: "admin/logs", methods: ["GET", "HEAD"]},
                "admin.root": {uri: "admin", methods: ["GET", "HEAD"]},
                "admin.forum.forum-covers.index": {uri: "admin/forum/forum-covers", methods: ["GET", "HEAD"]},
                "admin.forum.forum-covers.store": {uri: "admin/forum/forum-covers", methods: ["POST"]},
                "admin.forum.forum-covers.update": {
                    uri: "admin/forum/forum-covers/{forum_cover}",
                    methods: ["PUT", "PATCH"]
                },
                "artists.tracks.index": {uri: "beatmaps/artists/tracks", methods: ["GET", "HEAD"]},
                "artists.index": {uri: "beatmaps/artists", methods: ["GET", "HEAD"]},
                "artists.show": {uri: "beatmaps/artists/{artist}", methods: ["GET", "HEAD"]},
                "tracks.show": {uri: "beatmaps/artists/tracks/{track}", methods: ["GET", "HEAD"]},
                "packs.index": {uri: "beatmaps/packs", methods: ["GET", "HEAD"]},
                "packs.show": {uri: "beatmaps/packs/{pack}", methods: ["GET", "HEAD"]},
                "packs.raw": {uri: "beatmaps/packs/{pack}/raw", methods: ["GET", "HEAD"]},
                "beatmaps.": {uri: "beatmaps/{beatmap}/scores/users/{user}", methods: ["GET", "HEAD"]},
                "beatmaps.scores": {uri: "beatmaps/{beatmap}/scores", methods: ["GET", "HEAD"]},
                "beatmaps.update-owner": {uri: "beatmaps/{beatmap}/update-owner", methods: ["PUT"]},
                "beatmaps.show": {uri: "beatmaps/{beatmap}", methods: ["GET", "HEAD"]},
                "redirect:beatmapsets.discussions.index": {
                    uri: "beatmapsets/beatmap-discussions",
                    methods: ["GET", "HEAD"]
                },
                "redirect:beatmapsets.discussions.destroy": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}",
                    methods: ["DELETE"]
                },
                "redirect:beatmapsets.discussions.show": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}",
                    methods: ["GET", "HEAD"]
                },
                "redirect:beatmapsets.discussions.vote": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}/vote",
                    methods: ["PUT"]
                },
                "redirect:beatmapsets.discussions.restore": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}/restore",
                    methods: ["POST"]
                },
                "redirect:beatmapsets.discussions.deny-kudosu": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}/deny-kudosu",
                    methods: ["POST"]
                },
                "redirect:beatmapsets.discussions.allow-kudosu": {
                    uri: "beatmapsets/beatmap-discussions/{beatmap_discussion}/allow-kudosu",
                    methods: ["POST"]
                },
                "redirect:beatmapsets.discussions.posts.destroy": {
                    uri: "beatmapsets/beatmap-discussion-posts/{beatmap_discussion_post}",
                    methods: ["DELETE"]
                },
                "redirect:beatmapsets.discussions.posts.store": {
                    uri: "beatmapsets/beatmap-discussion-posts/{beatmap_discussion_post}",
                    methods: ["POST"]
                },
                "redirect:beatmapsets.discussions.posts.update": {
                    uri: "beatmapsets/beatmap-discussion-posts/{beatmap_discussion_post}",
                    methods: ["PUT"]
                },
                "redirect:beatmapsets.discussions.posts.restore": {
                    uri: "beatmapsets/beatmap-discussion-posts/{beatmap_discussion_post}/restore",
                    methods: ["POST"]
                },
                "redirect:beatmapsets.discussions.posts.index": {
                    uri: "beatmapsets/beatmap-discussion-posts",
                    methods: ["GET", "HEAD"]
                },
                "beatmapsets.events.index": {uri: "beatmapsets/events", methods: ["GET", "HEAD"]},
                "beatmapsets.redirect:follows.index": {uri: "beatmapsets/watches", methods: ["GET", "HEAD"]},
                "beatmapsets.watches.update": {uri: "beatmapsets/watches/{watch}", methods: ["PUT", "PATCH"]},
                "beatmapsets.watches.destroy": {uri: "beatmapsets/watches/{watch}", methods: ["DELETE"]},
                "beatmapsets.discussions.vote": {uri: "beatmapsets/discussions/{discussion}/vote", methods: ["PUT"]},
                "beatmapsets.discussions.restore": {
                    uri: "beatmapsets/discussions/{discussion}/restore",
                    methods: ["POST"]
                },
                "beatmapsets.discussions.deny-kudosu": {
                    uri: "beatmapsets/discussions/{discussion}/deny-kudosu",
                    methods: ["POST"]
                },
                "beatmapsets.discussions.allow-kudosu": {
                    uri: "beatmapsets/discussions/{discussion}/allow-kudosu",
                    methods: ["POST"]
                },
                "beatmapsets.discussions.posts.restore": {
                    uri: "beatmapsets/discussions/posts/{post}/restore",
                    methods: ["POST"]
                },
                "beatmapsets.discussions.posts.index": {uri: "beatmapsets/discussions/posts", methods: ["GET", "HEAD"]},
                "beatmapsets.discussions.posts.store": {uri: "beatmapsets/discussions/posts", methods: ["POST"]},
                "beatmapsets.discussions.posts.show": {
                    uri: "beatmapsets/discussions/posts/{post}",
                    methods: ["GET", "HEAD"]
                },
                "beatmapsets.discussions.posts.update": {
                    uri: "beatmapsets/discussions/posts/{post}",
                    methods: ["PUT", "PATCH"]
                },
                "beatmapsets.discussions.posts.destroy": {
                    uri: "beatmapsets/discussions/posts/{post}",
                    methods: ["DELETE"]
                },
                "beatmapsets.discussions.votes.index": {uri: "beatmapsets/discussions/votes", methods: ["GET", "HEAD"]},
                "beatmapsets.discussions.index": {uri: "beatmapsets/discussions", methods: ["GET", "HEAD"]},
                "beatmapsets.discussions.show": {uri: "beatmapsets/discussions/{discussion}", methods: ["GET", "HEAD"]},
                "beatmapsets.discussions.destroy": {uri: "beatmapsets/discussions/{discussion}", methods: ["DELETE"]},
                "beatmapsets.favourites.store": {uri: "beatmapsets/{beatmapset}/favourites", methods: ["POST"]},
                "beatmapsets.search": {uri: "beatmapsets/search/{filters?}", methods: ["GET", "HEAD"]},
                "beatmapsets.discussion": {
                    uri: "beatmapsets/{beatmapset}/discussion/{beatmap?}/{mode?}/{filter?}",
                    methods: ["GET", "HEAD"]
                },
                "beatmapsets.discussion.review": {uri: "beatmapsets/{beatmapset}/discussion/review", methods: ["POST"]},
                "beatmapsets.discussion-lock": {uri: "beatmapsets/{beatmapset}/discussion-lock", methods: ["POST"]},
                "beatmapsets.discussion-unlock": {uri: "beatmapsets/{beatmapset}/discussion-unlock", methods: ["POST"]},
                "beatmapsets.download": {uri: "beatmapsets/{beatmapset}/download", methods: ["GET", "HEAD"]},
                "beatmapsets.love": {uri: "beatmapsets/{beatmapset}/love", methods: ["PUT"]},
                "beatmapsets.remove-from-loved": {uri: "beatmapsets/{beatmapset}/love", methods: ["DELETE"]},
                "beatmapsets.nominate": {uri: "beatmapsets/{beatmapset}/nominate", methods: ["PUT"]},
                "beatmapsets.index": {uri: "beatmapsets", methods: ["GET", "HEAD"]},
                "beatmapsets.show": {uri: "beatmapsets/{beatmapset}", methods: ["GET", "HEAD"]},
                "beatmapsets.update": {uri: "beatmapsets/{beatmapset}", methods: ["PUT", "PATCH"]},
                "beatmapsets.destroy": {uri: "beatmapsets/{beatmapset}", methods: ["DELETE"]},
                "scores.download": {uri: "scores/{mode}/{score}/download", methods: ["GET", "HEAD"]},
                "scores.show": {uri: "scores/{mode}/{score}", methods: ["GET", "HEAD"]},
                "score-pins.destroy": {uri: "score-pins", methods: ["DELETE"]},
                "score-pins.reorder": {uri: "score-pins", methods: ["PUT"]},
                "score-pins.store": {uri: "score-pins", methods: ["POST"]},
                "client-verifications.create": {uri: "client-verifications/create", methods: ["GET", "HEAD"]},
                "client-verifications.store": {uri: "client-verifications", methods: ["POST"]},
                "comments.index": {uri: "comments", methods: ["GET", "HEAD"]},
                "comments.store": {uri: "comments", methods: ["POST"]},
                "comments.show": {uri: "comments/{comment}", methods: ["GET", "HEAD"]},
                "comments.update": {uri: "comments/{comment}", methods: ["PUT", "PATCH"]},
                "comments.destroy": {uri: "comments/{comment}", methods: ["DELETE"]},
                "comments.pin": {uri: "comments/{comment}/pin", methods: ["POST"]},
                "comments.restore": {uri: "comments/{comment}/restore", methods: ["POST"]},
                "comments.vote": {uri: "comments/{comment}/vote", methods: ["POST"]},
                "contests.index": {uri: "community/contests", methods: ["GET", "HEAD"]},
                "contests.show": {uri: "community/contests/{contest}", methods: ["GET", "HEAD"]},
                "contest-entries.vote": {uri: "community/contest-entries/{contest_entry}/vote", methods: ["PUT"]},
                "contest-entries.store": {uri: "community/contest-entries", methods: ["POST"]},
                "contest-entries.destroy": {uri: "community/contest-entries/{contest_entry}", methods: ["DELETE"]},
                "livestreams.promote": {uri: "community/livestreams/promote", methods: ["POST"]},
                "livestreams.index": {uri: "community/livestreams", methods: ["GET", "HEAD"]},
                "matches.show": {uri: "community/matches/{match}", methods: ["GET", "HEAD"]},
                "tournaments.unregister": {uri: "community/tournaments/{tournament}/unregister", methods: ["POST"]},
                "tournaments.register": {uri: "community/tournaments/{tournament}/register", methods: ["POST"]},
                "tournaments.index": {uri: "community/tournaments", methods: ["GET", "HEAD"]},
                "tournaments.show": {uri: "community/tournaments/{tournament}", methods: ["GET", "HEAD"]},
                "forum.forum-covers.store": {uri: "community/forums/forum-covers", methods: ["POST"]},
                "forum.forum-covers.update": {
                    uri: "community/forums/forum-covers/{forum_cover}",
                    methods: ["PUT", "PATCH"]
                },
                "forum.forum-covers.destroy": {uri: "community/forums/forum-covers/{forum_cover}", methods: ["DELETE"]},
                "forum.posts.raw": {uri: "community/forums/posts/{post}/raw", methods: ["GET", "HEAD"]},
                "forum.posts.restore": {uri: "community/forums/posts/{post}/restore", methods: ["POST"]},
                "forum.posts.show": {uri: "community/forums/posts/{post}", methods: ["GET", "HEAD"]},
                "forum.posts.edit": {uri: "community/forums/posts/{post}/edit", methods: ["GET", "HEAD"]},
                "forum.posts.update": {uri: "community/forums/posts/{post}", methods: ["PUT", "PATCH"]},
                "forum.posts.destroy": {uri: "community/forums/posts/{post}", methods: ["DELETE"]},
                "forum.topics.logs.index": {uri: "community/forums/topics/{topic}/logs", methods: ["GET", "HEAD"]},
                "forum.topics.edit-poll.store": {uri: "community/forums/topics/{topic}/edit-poll", methods: ["POST"]},
                "forum.topics.edit-poll": {uri: "community/forums/topics/{topic}/edit-poll", methods: ["GET", "HEAD"]},
                "forum.topics.issue-tag": {uri: "community/forums/topics/{topic}/issue-tag", methods: ["POST"]},
                "forum.topics.lock": {uri: "community/forums/topics/{topic}/lock", methods: ["POST"]},
                "forum.topics.move": {uri: "community/forums/topics/{topic}/move", methods: ["POST"]},
                "forum.topics.pin": {uri: "community/forums/topics/{topic}/pin", methods: ["POST"]},
                "forum.topics.reply": {uri: "community/forums/topics/{topic}/reply", methods: ["POST"]},
                "forum.topics.restore": {uri: "community/forums/topics/{topic}/restore", methods: ["POST"]},
                "forum.topics.vote": {uri: "community/forums/topics/{topic}/vote", methods: ["POST"]},
                "forum.topics.vote-feature": {uri: "community/forums/topics/{topic}/vote-feature", methods: ["POST"]},
                "forum.preview": {uri: "community/forums/topics/preview", methods: ["POST"]},
                "forum.topics.create": {uri: "community/forums/topics/create", methods: ["GET", "HEAD"]},
                "forum.topics.store": {uri: "community/forums/topics", methods: ["POST"]},
                "forum.topics.show": {uri: "community/forums/topics/{topic}", methods: ["GET", "HEAD"]},
                "forum.topics.update": {uri: "community/forums/topics/{topic}", methods: ["PUT", "PATCH"]},
                "forum.topics.destroy": {uri: "community/forums/topics/{topic}", methods: ["DELETE"]},
                "forum.topic-covers.store": {uri: "community/forums/topic-covers", methods: ["POST"]},
                "forum.topic-covers.update": {
                    uri: "community/forums/topic-covers/{topic_cover}",
                    methods: ["PUT", "PATCH"]
                },
                "forum.topic-covers.destroy": {uri: "community/forums/topic-covers/{topic_cover}", methods: ["DELETE"]},
                "forum.redirect:follows.index": {uri: "community/forums/topic-watches", methods: ["GET", "HEAD"]},
                "forum.topic-watches.update": {
                    uri: "community/forums/topic-watches/{topic_watch}",
                    methods: ["PUT", "PATCH"]
                },
                "forum.forums.mark-as-read": {uri: "community/forums/mark-as-read", methods: ["POST"]},
                "forum.forums.index": {uri: "community/forums", methods: ["GET", "HEAD"]},
                "forum.forums.show": {uri: "community/forums/{forum}", methods: ["GET", "HEAD"]},
                "chat.ack": {uri: "community/chat/ack", methods: ["POST"]},
                "chat.new": {uri: "community/chat/new", methods: ["POST"]},
                "chat.presence": {uri: "community/chat/presence", methods: ["GET", "HEAD"]},
                "chat.updates": {uri: "community/chat/updates", methods: ["GET", "HEAD"]},
                "chat.channels.messages.index": {
                    uri: "community/chat/channels/{channel}/messages",
                    methods: ["GET", "HEAD"]
                },
                "chat.channels.messages.store": {uri: "community/chat/channels/{channel}/messages", methods: ["POST"]},
                "chat.channels.join": {uri: "community/chat/channels/{channel}/users/{user}", methods: ["PUT"]},
                "chat.channels.part": {uri: "community/chat/channels/{channel}/users/{user}", methods: ["DELETE"]},
                "chat.channels.mark-as-read": {
                    uri: "community/chat/channels/{channel}/mark-as-read/{message}",
                    methods: ["PUT"]
                },
                "chat.channels.index": {uri: "community/chat/channels", methods: ["GET", "HEAD"]},
                "chat.channels.show": {uri: "community/chat/channels/{channel}", methods: ["GET", "HEAD"]},
                "chat.index": {uri: "community/chat", methods: ["GET", "HEAD"]},
                "groups.show": {uri: "groups/{group}", methods: ["GET", "HEAD"]},
                "account.edit": {uri: "home/account/edit", methods: ["GET", "HEAD"]},
                "account.avatar": {uri: "home/account/avatar", methods: ["POST"]},
                "account.cover": {uri: "home/account/cover", methods: ["POST"]},
                "account.email": {uri: "home/account/email", methods: ["PUT"]},
                "account.notification-options": {uri: "home/account/notification-options", methods: ["PUT"]},
                "account.options": {uri: "home/account/options", methods: ["PUT"]},
                "account.password": {uri: "home/account/password", methods: ["PUT"]},
                "account.reissue-code": {uri: "home/account/reissue-code", methods: ["POST"]},
                "account.sessions.destroy": {uri: "home/account/sessions/{session}", methods: ["DELETE"]},
                "account.": {uri: "home/account/verify", methods: ["GET", "HEAD"]},
                "account.verify": {uri: "home/account/verify", methods: ["POST"]},
                "account.update": {uri: "home/account", methods: ["PUT"]},
                "quick-search": {uri: "home/quick-search", methods: ["GET", "HEAD"]},
                search: {uri: "home/search", methods: ["GET", "HEAD"]},
                "bbcode-preview": {uri: "home/bbcode-preview", methods: ["POST"]},
                "changelog.build": {uri: "home/changelog/{stream}/{build}", methods: ["GET", "HEAD"]},
                "changelog.index": {uri: "home/changelog", methods: ["GET", "HEAD"]},
                "changelog.show": {uri: "home/changelog/{changelog}", methods: ["GET", "HEAD"]},
                download: {uri: "home/download", methods: ["GET", "HEAD"]},
                "set-locale": {uri: "home/set-locale", methods: ["POST"]},
                "support-the-game": {uri: "home/support", methods: ["GET", "HEAD"]},
                testflight: {uri: "home/testflight", methods: ["GET", "HEAD"]},
                "password-reset": {uri: "home/password-reset", methods: ["GET", "HEAD"]},
                "download-quota-check": {uri: "home/download-quota-check", methods: ["GET", "HEAD"]},
                "blocks.store": {uri: "home/blocks", methods: ["POST"]},
                "blocks.destroy": {uri: "home/blocks/{block}", methods: ["DELETE"]},
                "friends.index": {uri: "home/friends", methods: ["GET", "HEAD"]},
                "friends.store": {uri: "home/friends", methods: ["POST"]},
                "friends.destroy": {uri: "home/friends/{friend}", methods: ["DELETE"]},
                "news.index": {uri: "home/news", methods: ["GET", "HEAD"]},
                "news.store": {uri: "home/news", methods: ["POST"]},
                "news.show": {uri: "home/news/{news}", methods: ["GET", "HEAD"]},
                "news.update": {uri: "home/news/{news}", methods: ["PUT", "PATCH"]},
                "messages.users.show": {uri: "home/messages/users/{user}", methods: ["GET", "HEAD"]},
                "follows.store": {uri: "home/follows", methods: ["POST"]},
                "follows.index": {uri: "home/follows/{subtype?}", methods: ["GET", "HEAD"]},
                "follows.destroy": {uri: "home/follows", methods: ["DELETE"]},
                "notifications.index": {uri: "notifications", methods: ["GET", "HEAD"]},
                "notifications.endpoint": {uri: "notifications/endpoint", methods: ["GET", "HEAD"]},
                "notifications.mark-read": {uri: "notifications/mark-read", methods: ["POST"]},
                legal: {uri: "legal/{locale?}/{path?}", methods: ["GET", "HEAD"]},
                "multiplayer.rooms.show": {uri: "multiplayer/rooms/{room}", methods: ["GET", "HEAD"]},
                "oauth.authorized-clients.destroy": {
                    uri: "oauth/authorized-clients/{authorized_client}",
                    methods: ["DELETE"]
                },
                "oauth.clients.index": {uri: "oauth/clients", methods: ["GET", "HEAD"]},
                "oauth.clients.store": {uri: "oauth/clients", methods: ["POST"]},
                "oauth.clients.update": {uri: "oauth/clients/{client}", methods: ["PUT", "PATCH"]},
                "oauth.clients.destroy": {uri: "oauth/clients/{client}", methods: ["DELETE"]},
                "oauth.clients.reset-secret": {uri: "oauth/clients/{client}/reset-secret", methods: ["POST"]},
                rankings: {uri: "rankings/{mode?}/{type?}", methods: ["GET", "HEAD"]},
                "reports.store": {uri: "reports", methods: ["POST"]},
                login: {uri: "session", methods: ["POST"]},
                logout: {uri: "session", methods: ["DELETE"]},
                "users.check-username-availability": {uri: "users/check-username-availability", methods: ["POST"]},
                "users.check-username-exists": {uri: "users/check-username-exists", methods: ["POST"]},
                "users.disabled": {uri: "users/disabled", methods: ["GET", "HEAD"]},
                "users.card": {uri: "users/{user}/card", methods: ["GET", "HEAD"]},
                "users.page": {uri: "users/{user}/page", methods: ["PUT"]},
                "users.multiplayer.index": {uri: "users/{user}/{typeGroup}", methods: ["GET", "HEAD"]},
                "users.modding.index": {uri: "users/{user}/modding", methods: ["GET", "HEAD"]},
                "users.modding.posts": {uri: "users/{user}/modding/posts", methods: ["GET", "HEAD"]},
                "users.modding.votes-given": {uri: "users/{user}/modding/votes-given", methods: ["GET", "HEAD"]},
                "users.modding.votes-received": {uri: "users/{user}/modding/votes-received", methods: ["GET", "HEAD"]},
                "users.kudosu": {uri: "users/{user}/kudosu", methods: ["GET", "HEAD"]},
                "users.recent-activity": {uri: "users/{user}/recent_activity", methods: ["GET", "HEAD"]},
                "users.scores": {uri: "users/{user}/scores/{type}", methods: ["GET", "HEAD"]},
                "users.beatmapsets": {uri: "users/{user}/beatmapsets/{type}", methods: ["GET", "HEAD"]},
                "users.posts": {uri: "users/{user}/posts", methods: ["GET", "HEAD"]},
                "users.show": {uri: "users/{user}/{mode?}", methods: ["GET", "HEAD"]},
                "users.store": {uri: "users", methods: ["POST"]},
                "wiki.sitemap": {uri: "wiki/{locale}/Sitemap", methods: ["GET", "HEAD"]},
                "wiki.image": {uri: "wiki/images/{path}", methods: ["GET", "HEAD"]},
                "wiki.show": {uri: "wiki/{locale?}/{path?}", methods: ["GET", "HEAD"]},
                "wiki-suggestions": {uri: "wiki-suggestions", methods: ["GET", "HEAD"]},
                "store.redirect:store.products.index": {uri: "store", methods: ["GET", "HEAD"]},
                "store.products.index": {uri: "store/listing", methods: ["GET", "HEAD"]},
                "store.invoice.show": {uri: "store/invoice/{invoice}", methods: ["GET", "HEAD"]},
                "store.": {uri: "store/products/{product}/notification-request", methods: ["DELETE"]},
                "store.notification-request": {uri: "store/products/{product}/notification-request", methods: ["POST"]},
                "store.cart.show": {uri: "store/cart", methods: ["GET", "HEAD"]},
                "store.cart.store": {uri: "store/cart", methods: ["POST"]},
                "store.checkout.store": {uri: "store/checkout", methods: ["POST"]},
                "store.checkout.show": {uri: "store/checkout/{checkout}", methods: ["GET", "HEAD"]},
                "store.orders.index": {uri: "store/orders", methods: ["GET", "HEAD"]},
                "store.orders.destroy": {uri: "store/orders/{order}", methods: ["DELETE"]},
                "store.redirect:store.products.show": {uri: "store/product/{product}", methods: ["GET", "HEAD"]},
                "store.products.show": {uri: "store/products/{product}", methods: ["GET", "HEAD"]},
                "payments.paypal.approved": {uri: "payments/paypal/approved", methods: ["GET", "HEAD"]},
                "payments.paypal.declined": {uri: "payments/paypal/declined", methods: ["GET", "HEAD"]},
                "payments.paypal.create": {uri: "payments/paypal/create", methods: ["POST"]},
                "payments.paypal.completed": {uri: "payments/paypal/completed", methods: ["GET", "HEAD"]},
                "payments.paypal.ipn": {uri: "payments/paypal/ipn", methods: ["POST"]},
                "payments.xsolla.completed": {uri: "payments/xsolla/completed", methods: ["GET", "HEAD"]},
                "payments.xsolla.token": {uri: "payments/xsolla/token", methods: ["POST"]},
                "payments.xsolla.callback": {uri: "payments/xsolla/callback", methods: ["POST"]},
                "payments.centili.callback": {uri: "payments/centili/callback", methods: ["POST", "GET", "HEAD"]},
                "payments.centili.completed": {uri: "payments/centili/completed", methods: ["GET", "HEAD"]},
                "payments.centili.failed": {uri: "payments/centili/failed", methods: ["GET", "HEAD"]},
                "payments.shopify.callback": {uri: "payments/shopify/callback", methods: ["POST"]},
                home: {uri: "home", methods: ["GET", "HEAD"]},
                "redirect:home": {uri: "/", methods: ["GET", "HEAD"]},
                "redirect:forum.posts.show": {uri: "forum/p/{post}", methods: ["GET", "HEAD"]},
                "redirect:forum.posts.show:": {uri: "po/{post}", methods: ["GET", "HEAD"]},
                "redirect:forum.topics.show": {uri: "forum/t/{topic}", methods: ["GET", "HEAD"]},
                "redirect:forum.forums.show": {uri: "forum/{forum}", methods: ["GET", "HEAD"]},
                "redirect:beatmaps.show": {uri: "b/{beatmap}", methods: ["GET", "HEAD"]},
                "redirect:groups.show": {uri: "g/{group}", methods: ["GET", "HEAD"]},
                "redirect:beatmapsets.show": {uri: "s/{beatmapset}", methods: ["GET", "HEAD"]},
                "redirect:users.show": {uri: "u/{user}", methods: ["GET", "HEAD"]},
                "redirect:forum.forums.index": {uri: "forum", methods: ["GET", "HEAD"]},
                "redirect:matches.show": {uri: "mp/{match}", methods: ["GET", "HEAD"]},
                "redirect:wiki.show": {uri: "help/wiki/{path?}", methods: ["GET", "HEAD"]},
                "api.beatmaps.lookup": {uri: "api/v2/beatmaps/lookup", methods: ["GET", "HEAD"]},
                "api.beatmaps.": {uri: "api/v2/beatmaps/{beatmap}/scores/users/{user}/all", methods: ["GET", "HEAD"]},
                "api.beatmaps.scores": {uri: "api/v2/beatmaps/{beatmap}/scores", methods: ["GET", "HEAD"]},
                "api.beatmaps.solo.score-tokens.store": {
                    uri: "api/v2/beatmaps/{beatmap}/solo/scores",
                    methods: ["POST"]
                },
                "api.beatmaps.solo.scores.store": {
                    uri: "api/v2/beatmaps/{beatmap}/solo/scores/{token}",
                    methods: ["PUT"]
                },
                "api.beatmaps.index": {uri: "api/v2/beatmaps", methods: ["GET", "HEAD"]},
                "api.beatmaps.show": {uri: "api/v2/beatmaps/{beatmap}", methods: ["GET", "HEAD"]},
                "api.beatmaps.attributes": {uri: "api/v2/beatmaps/{beatmap}/attributes", methods: ["POST"]},
                "api.beatmapsets.events.index": {uri: "api/v2/beatmapsets/events", methods: ["GET", "HEAD"]},
                "api.beatmapsets.discussions.posts.index": {
                    uri: "api/v2/beatmapsets/discussions/posts",
                    methods: ["GET", "HEAD"]
                },
                "api.beatmapsets.discussions.votes.index": {
                    uri: "api/v2/beatmapsets/discussions/votes",
                    methods: ["GET", "HEAD"]
                },
                "api.beatmapsets.discussions.index": {uri: "api/v2/beatmapsets/discussions", methods: ["GET", "HEAD"]},
                "api.beatmapsets.favourites.store": {
                    uri: "api/v2/beatmapsets/{beatmapset}/favourites",
                    methods: ["POST"]
                },
                "api.comments.index": {uri: "api/v2/comments", methods: ["GET", "HEAD"]},
                "api.comments.store": {uri: "api/v2/comments", methods: ["POST"]},
                "api.comments.show": {uri: "api/v2/comments/{comment}", methods: ["GET", "HEAD"]},
                "api.comments.update": {uri: "api/v2/comments/{comment}", methods: ["PUT", "PATCH"]},
                "api.comments.destroy": {uri: "api/v2/comments/{comment}", methods: ["DELETE"]},
                "api.comments.vote": {uri: "api/v2/comments/{comment}/vote", methods: ["POST"]},
                "api.": {uri: "api/v2/users/{user}/recent_activity", methods: ["GET", "HEAD"]},
                "api.chat.new": {uri: "api/v2/chat/new", methods: ["POST"]},
                "api.chat.updates": {uri: "api/v2/chat/updates", methods: ["GET", "HEAD"]},
                "api.chat.presence": {uri: "api/v2/chat/presence", methods: ["GET", "HEAD"]},
                "api.chat.channels.messages.index": {
                    uri: "api/v2/chat/channels/{channel}/messages",
                    methods: ["GET", "HEAD"]
                },
                "api.chat.channels.messages.store": {uri: "api/v2/chat/channels/{channel}/messages", methods: ["POST"]},
                "api.chat.channels.join": {uri: "api/v2/chat/channels/{channel}/users/{user}", methods: ["PUT"]},
                "api.chat.channels.part": {uri: "api/v2/chat/channels/{channel}/users/{user}", methods: ["DELETE"]},
                "api.chat.channels.mark-as-read": {
                    uri: "api/v2/chat/channels/{channel}/mark-as-read/{message}",
                    methods: ["PUT"]
                },
                "api.chat.channels.index": {uri: "api/v2/chat/channels", methods: ["GET", "HEAD"]},
                "api.chat.channels.store": {uri: "api/v2/chat/channels", methods: ["POST"]},
                "api.chat.channels.show": {uri: "api/v2/chat/channels/{channel}", methods: ["GET", "HEAD"]},
                "api.changelog.build": {uri: "api/v2/changelog/{stream}/{build}", methods: ["GET", "HEAD"]},
                "api.changelog.index": {uri: "api/v2/changelog", methods: ["GET", "HEAD"]},
                "api.changelog.show": {uri: "api/v2/changelog/{changelog}", methods: ["GET", "HEAD"]},
                "api.forum.topics.reply": {uri: "api/v2/forums/topics/{topic}/reply", methods: ["POST"]},
                "api.forum.topics.store": {uri: "api/v2/forums/topics", methods: ["POST"]},
                "api.forum.topics.show": {uri: "api/v2/forums/topics/{topic}", methods: ["GET", "HEAD"]},
                "api.forum.topics.update": {uri: "api/v2/forums/topics/{topic}", methods: ["PUT", "PATCH"]},
                "api.forum.posts.update": {uri: "api/v2/forums/posts/{post}", methods: ["PUT", "PATCH"]},
                "api.matches.index": {uri: "api/v2/matches", methods: ["GET", "HEAD"]},
                "api.matches.show": {uri: "api/v2/matches/{match}", methods: ["GET", "HEAD"]},
                "api.rooms.index": {uri: "api/v2/rooms/{mode?}", methods: ["GET", "HEAD"]},
                "api.rooms.join": {uri: "api/v2/rooms/{room}/users/{user}", methods: ["PUT"]},
                "api.rooms.part": {uri: "api/v2/rooms/{room}/users/{user}", methods: ["DELETE"]},
                "api.rooms.": {uri: "api/v2/rooms/{room}/leaderboard", methods: ["GET", "HEAD"]},
                "api.rooms.playlist.": {
                    uri: "api/v2/rooms/{room}/playlist/{playlist}/scores/users/{user}",
                    methods: ["GET", "HEAD"]
                },
                "api.rooms.playlist.scores.index": {
                    uri: "api/v2/rooms/{room}/playlist/{playlist}/scores",
                    methods: ["GET", "HEAD"]
                },
                "api.rooms.playlist.scores.store": {
                    uri: "api/v2/rooms/{room}/playlist/{playlist}/scores",
                    methods: ["POST"]
                },
                "api.rooms.playlist.scores.show": {
                    uri: "api/v2/rooms/{room}/playlist/{playlist}/scores/{score}",
                    methods: ["GET", "HEAD"]
                },
                "api.rooms.playlist.scores.update": {
                    uri: "api/v2/rooms/{room}/playlist/{playlist}/scores/{score}",
                    methods: ["PUT", "PATCH"]
                },
                "api.reports.store": {uri: "api/v2/reports", methods: ["POST"]},
                "api.rooms.store": {uri: "api/v2/rooms", methods: ["POST"]},
                "api.rooms.show": {uri: "api/v2/rooms/{room}", methods: ["GET", "HEAD"]},
                "api.seasonal-backgrounds.index": {uri: "api/v2/seasonal-backgrounds", methods: ["GET", "HEAD"]},
                "api.scores.download": {uri: "api/v2/scores/{mode}/{score}/download", methods: ["GET", "HEAD"]},
                "api.scores.show": {uri: "api/v2/scores/{mode}/{score}", methods: ["GET", "HEAD"]},
                "api.beatmapsets.show": {uri: "api/v2/beatmapsets/{beatmapset}", methods: ["GET", "HEAD"]},
                "api.friends.index": {uri: "api/v2/friends", methods: ["GET", "HEAD"]},
                "api.me": {uri: "api/v2/me/{mode?}", methods: ["GET", "HEAD"]},
                "api.oauth.tokens.current": {uri: "api/v2/oauth/tokens/current", methods: ["DELETE"]},
                "api.news.index": {uri: "api/v2/news", methods: ["GET", "HEAD"]},
                "api.news.show": {uri: "api/v2/news/{news}", methods: ["GET", "HEAD"]},
                "api.notifications.index": {uri: "api/v2/notifications", methods: ["GET", "HEAD"]},
                "api.notifications.mark-read": {uri: "api/v2/notifications/mark-read", methods: ["POST"]},
                "api.spotlights.index": {uri: "api/v2/spotlights", methods: ["GET", "HEAD"]},
                "api.users.show": {uri: "api/v2/users/{user}/{mode?}", methods: ["GET", "HEAD"]},
                "api.users.index": {uri: "api/v2/users", methods: ["GET", "HEAD"]},
                "api.wiki.show": {uri: "api/v2/wiki/{locale}/{path}", methods: ["GET", "HEAD"]},
                "interop.": {uri: "_lio/artist-tracks/reindex-all", methods: ["POST"]},
                "interop.user-achievement": {
                    uri: "_lio/user-achievement/{user}/{achievement}/{beatmap?}",
                    methods: ["POST"]
                },
                "interop.users.store": {uri: "_lio/users", methods: ["POST"]},
                "interop.beatmapsets.broadcast-new": {
                    uri: "_lio/beatmapsets/{beatmapset}/broadcast-new",
                    methods: ["POST"]
                },
                "interop.beatmapsets.broadcast-revive": {
                    uri: "_lio/beatmapsets/{beatmapset}/broadcast-revive",
                    methods: ["POST"]
                },
                "interop.beatmapsets.disqualify": {uri: "_lio/beatmapsets/{beatmapset}/disqualify", methods: ["POST"]},
                "interop.beatmapsets.destroy": {uri: "_lio/beatmapsets/{beatmapset}", methods: ["DELETE"]},
                "interop.indexing.bulk.store": {uri: "_lio/indexing/bulk", methods: ["POST"]},
                "interop.user-group.update": {uri: "_lio/users/{user}/groups/{group}", methods: ["PUT"]},
                "interop.user-group.destroy": {uri: "_lio/users/{user}/groups/{group}", methods: ["DELETE"]},
                "interop.user-group.set-default": {uri: "_lio/users/{user}/groups/{group}/default", methods: ["POST"]},
                "oauth.passport.token": {uri: "oauth/token", methods: ["POST"]},
                "oauth.authorizations.authorize": {uri: "oauth/authorize", methods: ["GET", "HEAD"]},
                "oauth.": {uri: "oauth/authorize", methods: ["DELETE"]}
            }
        };

        function i() {
            return (i = Object.assign || function (e) {
                for (var t = 1; t < arguments.length; t++) {
                    var s = arguments[t];
                    for (var r in s) Object.prototype.hasOwnProperty.call(s, r) && (e[r] = s[r])
                }
                return e
            }).apply(this, arguments)
        }

        "undefined" != typeof window && void 0 !== window.Ziggy && Object.assign(n.routes, window.Ziggy.routes);
        var a = Object.prototype.hasOwnProperty, o = Array.isArray, l = function () {
            for (var e = [], t = 0; t < 256; ++t) e.push("%" + ((t < 16 ? "0" : "") + t.toString(16)).toUpperCase());
            return e
        }(), c = function (e, t) {
            for (var s = t && t.plainObjects ? Object.create(null) : {}, r = 0; r < e.length; ++r) void 0 !== e[r] && (s[r] = e[r]);
            return s
        }, u = {
            arrayToObject: c, assign: function (e, t) {
                return Object.keys(t).reduce((function (e, s) {
                    return e[s] = t[s], e
                }), e)
            }, combine: function (e, t) {
                return [].concat(e, t)
            }, compact: function (e) {
                for (var t = [{
                    obj: {o: e},
                    prop: "o"
                }], s = [], r = 0; r < t.length; ++r) for (var n = t[r], i = n.obj[n.prop], a = Object.keys(i), l = 0; l < a.length; ++l) {
                    var c = a[l], u = i[c];
                    "object" == typeof u && null !== u && -1 === s.indexOf(u) && (t.push({obj: i, prop: c}), s.push(u))
                }
                return function (e) {
                    for (; e.length > 1;) {
                        var t = e.pop(), s = t.obj[t.prop];
                        if (o(s)) {
                            for (var r = [], n = 0; n < s.length; ++n) void 0 !== s[n] && r.push(s[n]);
                            t.obj[t.prop] = r
                        }
                    }
                }(t), e
            }, decode: function (e, t, s) {
                var r = e.replace(/\+/g, " ");
                if ("iso-8859-1" === s) return r.replace(/%[0-9a-f]{2}/gi, unescape);
                try {
                    return decodeURIComponent(r)
                } catch (e) {
                    return r
                }
            }, encode: function (e, t, s) {
                if (0 === e.length) return e;
                var r = e;
                if ("symbol" == typeof e ? r = Symbol.prototype.toString.call(e) : "string" != typeof e && (r = String(e)), "iso-8859-1" === s) return escape(r).replace(/%u[0-9a-f]{4}/gi, (function (e) {
                    return "%26%23" + parseInt(e.slice(2), 16) + "%3B"
                }));
                for (var n = "", i = 0; i < r.length; ++i) {
                    var a = r.charCodeAt(i);
                    45 === a || 46 === a || 95 === a || 126 === a || a >= 48 && a <= 57 || a >= 65 && a <= 90 || a >= 97 && a <= 122 ? n += r.charAt(i) : a < 128 ? n += l[a] : a < 2048 ? n += l[192 | a >> 6] + l[128 | 63 & a] : a < 55296 || a >= 57344 ? n += l[224 | a >> 12] + l[128 | a >> 6 & 63] + l[128 | 63 & a] : (a = 65536 + ((1023 & a) << 10 | 1023 & r.charCodeAt(i += 1)), n += l[240 | a >> 18] + l[128 | a >> 12 & 63] + l[128 | a >> 6 & 63] + l[128 | 63 & a])
                }
                return n
            }, isBuffer: function (e) {
                return !(!e || "object" != typeof e || !(e.constructor && e.constructor.isBuffer && e.constructor.isBuffer(e)))
            }, isRegExp: function (e) {
                return "[object RegExp]" === Object.prototype.toString.call(e)
            }, maybeMap: function (e, t) {
                if (o(e)) {
                    for (var s = [], r = 0; r < e.length; r += 1) s.push(t(e[r]));
                    return s
                }
                return t(e)
            }, merge: function e(t, s, r) {
                if (!s) return t;
                if ("object" != typeof s) {
                    if (o(t)) t.push(s); else {
                        if (!t || "object" != typeof t) return [t, s];
                        (r && (r.plainObjects || r.allowPrototypes) || !a.call(Object.prototype, s)) && (t[s] = !0)
                    }
                    return t
                }
                if (!t || "object" != typeof t) return [t].concat(s);
                var n = t;
                return o(t) && !o(s) && (n = c(t, r)), o(t) && o(s) ? (s.forEach((function (s, n) {
                    if (a.call(t, n)) {
                        var i = t[n];
                        i && "object" == typeof i && s && "object" == typeof s ? t[n] = e(i, s, r) : t.push(s)
                    } else t[n] = s
                })), t) : Object.keys(s).reduce((function (t, n) {
                    var i = s[n];
                    return t[n] = a.call(t, n) ? e(t[n], i, r) : i, t
                }), n)
            }
        }, d = String.prototype.replace, p = /%20/g, m = {RFC1738: "RFC1738", RFC3986: "RFC3986"}, h = u.assign({
            default: m.RFC3986, formatters: {
                RFC1738: function (e) {
                    return d.call(e, p, "+")
                }, RFC3986: function (e) {
                    return String(e)
                }
            }
        }, m), b = Object.prototype.hasOwnProperty, f = {
            brackets: function (e) {
                return e + "[]"
            }, comma: "comma", indices: function (e, t) {
                return e + "[" + t + "]"
            }, repeat: function (e) {
                return e
            }
        }, v = Array.isArray, g = Array.prototype.push, y = function (e, t) {
            g.apply(e, v(t) ? t : [t])
        }, w = Date.prototype.toISOString, _ = h.default, O = {
            addQueryPrefix: !1,
            allowDots: !1,
            charset: "utf-8",
            charsetSentinel: !1,
            delimiter: "&",
            encode: !0,
            encoder: u.encode,
            encodeValuesOnly: !1,
            format: _,
            formatter: h.formatters[_],
            indices: !1,
            serializeDate: function (e) {
                return w.call(e)
            },
            skipNulls: !1,
            strictNullHandling: !1
        }, j = function e(t, s, r, n, i, a, o, l, c, d, p, m, h) {
            var b, f = t;
            if ("function" == typeof o ? f = o(s, f) : f instanceof Date ? f = d(f) : "comma" === r && v(f) && (f = u.maybeMap(f, (function (e) {
                return e instanceof Date ? d(e) : e
            })).join(",")), null === f) {
                if (n) return a && !m ? a(s, O.encoder, h, "key") : s;
                f = ""
            }
            if ("string" == typeof (b = f) || "number" == typeof b || "boolean" == typeof b || "symbol" == typeof b || "bigint" == typeof b || u.isBuffer(f)) return a ? [p(m ? s : a(s, O.encoder, h, "key")) + "=" + p(a(f, O.encoder, h, "value"))] : [p(s) + "=" + p(String(f))];
            var g, w = [];
            if (void 0 === f) return w;
            if (v(o)) g = o; else {
                var _ = Object.keys(f);
                g = l ? _.sort(l) : _
            }
            for (var j = 0; j < g.length; ++j) {
                var E = g[j], P = f[E];
                if (!i || null !== P) {
                    var k = v(f) ? "function" == typeof r ? r(s, E) : s : s + (c ? "." + E : "[" + E + "]");
                    y(w, e(P, k, r, n, i, a, o, l, c, d, p, m, h))
                }
            }
            return w
        }, E = Object.prototype.hasOwnProperty, P = Array.isArray, k = {
            allowDots: !1,
            allowPrototypes: !1,
            arrayLimit: 20,
            charset: "utf-8",
            charsetSentinel: !1,
            comma: !1,
            decoder: u.decode,
            delimiter: "&",
            depth: 5,
            ignoreQueryPrefix: !1,
            interpretNumericEntities: !1,
            parameterLimit: 1e3,
            parseArrays: !0,
            plainObjects: !1,
            strictNullHandling: !1
        }, N = function (e) {
            return e.replace(/&#(\d+);/g, (function (e, t) {
                return String.fromCharCode(parseInt(t, 10))
            }))
        }, S = function (e, t) {
            return e && "string" == typeof e && t.comma && e.indexOf(",") > -1 ? e.split(",") : e
        }, T = function (e, t, s, r) {
            if (e) {
                var n = s.allowDots ? e.replace(/\.([^.[]+)/g, "[$1]") : e, i = /(\[[^[\]]*])/g,
                    a = s.depth > 0 && /(\[[^[\]]*])/.exec(n), o = a ? n.slice(0, a.index) : n, l = [];
                if (o) {
                    if (!s.plainObjects && E.call(Object.prototype, o) && !s.allowPrototypes) return;
                    l.push(o)
                }
                for (var c = 0; s.depth > 0 && null !== (a = i.exec(n)) && c < s.depth;) {
                    if (c += 1, !s.plainObjects && E.call(Object.prototype, a[1].slice(1, -1)) && !s.allowPrototypes) return;
                    l.push(a[1])
                }
                return a && l.push("[" + n.slice(a.index) + "]"), function (e, t, s, r) {
                    for (var n = r ? t : S(t, s), i = e.length - 1; i >= 0; --i) {
                        var a, o = e[i];
                        if ("[]" === o && s.parseArrays) a = [].concat(n); else {
                            a = s.plainObjects ? Object.create(null) : {};
                            var l = "[" === o.charAt(0) && "]" === o.charAt(o.length - 1) ? o.slice(1, -1) : o,
                                c = parseInt(l, 10);
                            s.parseArrays || "" !== l ? !isNaN(c) && o !== l && String(c) === l && c >= 0 && s.parseArrays && c <= s.arrayLimit ? (a = [])[c] = n : a[l] = n : a = {0: n}
                        }
                        n = a
                    }
                    return n
                }(l, t, s, r)
            }
        };

        class C {
            constructor(e, t, s) {
                var r;
                this.name = e, this.definition = t, this.bindings = null != (r = t.bindings) ? r : {}, this.config = s
            }

            get template() {
                return `${this.config.absolute ? this.definition.domain ? `${this.config.url.match(/^\w+:\/\//)[0]}${this.definition.domain}${this.config.port ? ":" + this.config.port : ""}` : this.config.url : ""}/${this.definition.uri}`.replace(/\/+$/, "")
            }

            get parameterSegments() {
                var e, t;
                return null != (e = null === (t = this.template.match(/{[^}?]+\??}/g)) || void 0 === t ? void 0 : t.map(e => ({
                    name: e.replace(/{|\??}/g, ""),
                    required: !/\?}$/.test(e)
                }))) ? e : []
            }

            matchesUrl(e) {
                if (!this.definition.methods.includes("GET")) return !1;
                const t = this.template.replace(/\/{[^}?]*\?}/g, "(/[^/?]+)?").replace(/{[^}]+}/g, "[^/?]+").replace(/^\w+:\/\//, "");
                return new RegExp(`^${t}$`).test(e.replace(/\/+$/, "").split("?").shift())
            }

            compile(e) {
                return this.parameterSegments.length ? this.template.replace(/{([^}?]+)\??}/g, (t, s) => {
                    var r;
                    if ([null, void 0].includes(e[s]) && this.parameterSegments.find(({name: e}) => e === s).required) throw new Error(`Ziggy error: '${s}' parameter is required for route '${this.name}'.`);
                    return encodeURIComponent(null != (r = e[s]) ? r : "")
                }).replace(/\/+$/, "") : this.template
            }
        }

        class x extends String {
            constructor(e, t, s = !0, r) {
                var n;
                if (super(), this.t = null != (n = null != r ? r : Ziggy) ? n : null === globalThis || void 0 === globalThis ? void 0 : globalThis.Ziggy, this.t = i({}, this.t, {absolute: s}), e) {
                    if (!this.t.routes[e]) throw new Error(`Ziggy error: route '${e}' is not in the route list.`);
                    this.i = new C(e, this.t.routes[e], this.t), this.u = this.s(t)
                }
            }

            toString() {
                const e = Object.keys(this.u).filter(e => !this.i.parameterSegments.some(({name: t}) => t === e)).filter(e => "_query" !== e).reduce((e, t) => i({}, e, {[t]: this.u[t]}), {});
                return this.i.compile(this.u) + function (e, t) {
                    var s, r = e, n = function (e) {
                        if (!e) return O;
                        if (null != e.encoder && "function" != typeof e.encoder) throw new TypeError("Encoder has to be a function.");
                        var t = e.charset || O.charset;
                        if (void 0 !== e.charset && "utf-8" !== e.charset && "iso-8859-1" !== e.charset) throw new TypeError("The charset option must be either utf-8, iso-8859-1, or undefined");
                        var s = h.default;
                        if (void 0 !== e.format) {
                            if (!b.call(h.formatters, e.format)) throw new TypeError("Unknown format option provided.");
                            s = e.format
                        }
                        var r = h.formatters[s], n = O.filter;
                        return ("function" == typeof e.filter || v(e.filter)) && (n = e.filter), {
                            addQueryPrefix: "boolean" == typeof e.addQueryPrefix ? e.addQueryPrefix : O.addQueryPrefix,
                            allowDots: void 0 === e.allowDots ? O.allowDots : !!e.allowDots,
                            charset: t,
                            charsetSentinel: "boolean" == typeof e.charsetSentinel ? e.charsetSentinel : O.charsetSentinel,
                            delimiter: void 0 === e.delimiter ? O.delimiter : e.delimiter,
                            encode: "boolean" == typeof e.encode ? e.encode : O.encode,
                            encoder: "function" == typeof e.encoder ? e.encoder : O.encoder,
                            encodeValuesOnly: "boolean" == typeof e.encodeValuesOnly ? e.encodeValuesOnly : O.encodeValuesOnly,
                            filter: n,
                            formatter: r,
                            serializeDate: "function" == typeof e.serializeDate ? e.serializeDate : O.serializeDate,
                            skipNulls: "boolean" == typeof e.skipNulls ? e.skipNulls : O.skipNulls,
                            sort: "function" == typeof e.sort ? e.sort : null,
                            strictNullHandling: "boolean" == typeof e.strictNullHandling ? e.strictNullHandling : O.strictNullHandling
                        }
                    }(t);
                    "function" == typeof n.filter ? r = (0, n.filter)("", r) : v(n.filter) && (s = n.filter);
                    var i = [];
                    if ("object" != typeof r || null === r) return "";
                    var a = f[t && t.arrayFormat in f ? t.arrayFormat : t && "indices" in t ? t.indices ? "indices" : "repeat" : "indices"];
                    s || (s = Object.keys(r)), n.sort && s.sort(n.sort);
                    for (var o = 0; o < s.length; ++o) {
                        var l = s[o];
                        n.skipNulls && null === r[l] || y(i, j(r[l], l, a, n.strictNullHandling, n.skipNulls, n.encode ? n.encoder : null, n.filter, n.sort, n.allowDots, n.serializeDate, n.formatter, n.encodeValuesOnly, n.charset))
                    }
                    var c = i.join(n.delimiter), u = !0 === n.addQueryPrefix ? "?" : "";
                    return n.charsetSentinel && (u += "iso-8859-1" === n.charset ? "utf8=%26%2310003%3B&" : "utf8=%E2%9C%93&"), c.length > 0 ? u + c : ""
                }(i({}, e, this.u._query), {
                    addQueryPrefix: !0,
                    arrayFormat: "indices",
                    encodeValuesOnly: !0,
                    skipNulls: !0,
                    encoder: (e, t) => "boolean" == typeof e ? Number(e) : t(e)
                })
            }

            current(e, t) {
                const s = this.t.absolute ? this.l.host + this.l.pathname : this.l.pathname.replace(this.t.url.replace(/^\w*:\/\/[^/]+/, ""), "").replace(/^\/+/, "/"), [r, n] = Object.entries(this.t.routes).find(([t, r]) => new C(e, r, this.t).matchesUrl(s)) || [void 0, void 0];
                if (!e) return r;
                const i = new RegExp(`^${e.replace(".","\\.").replace("*",".*")}$`).test(r);
                if ([null, void 0].includes(t) || !i) return i;
                const a = new C(r, n, this.t);
                t = this.s(t, a);
                const o = this.h(n);
                return !(!Object.values(t).every(e => !e) || Object.values(o).length) || Object.entries(t).every(([e, t]) => o[e] == t)
            }

            get l() {
                var e, t, s, r, n, i;
                const {
                    host: a = "",
                    pathname: o = "",
                    search: l = ""
                } = "undefined" != typeof window ? window.location : {};
                return {
                    host: null != (e = null === (t = this.t.location) || void 0 === t ? void 0 : t.host) ? e : a,
                    pathname: null != (s = null === (r = this.t.location) || void 0 === r ? void 0 : r.pathname) ? s : o,
                    search: null != (n = null === (i = this.t.location) || void 0 === i ? void 0 : i.search) ? n : l
                }
            }

            get params() {
                return this.h(this.t.routes[this.current()])
            }

            has(e) {
                return Object.keys(this.t.routes).includes(e)
            }

            s(e = {}, t = this.i) {
                e = ["string", "number"].includes(typeof e) ? [e] : e;
                const s = t.parameterSegments.filter(({name: e}) => !this.t.defaults[e]);
                return Array.isArray(e) ? e = e.reduce((e, t, r) => i({}, e, s[r] ? {[s[r].name]: t} : {[t]: ""}), {}) : 1 !== s.length || e[s[0].name] || !e.hasOwnProperty(Object.values(t.bindings)[0]) && !e.hasOwnProperty("id") || (e = {[s[0].name]: e}), i({}, this.p(t), this.v(e, t.bindings))
            }

            p(e) {
                return e.parameterSegments.filter(({name: e}) => this.t.defaults[e]).reduce((e, {name: t}, s) => i({}, e, {[t]: this.t.defaults[t]}), {})
            }

            v(e, t = {}) {
                return Object.entries(e).reduce((e, [s, r]) => {
                    if (!r || "object" != typeof r || Array.isArray(r) || "_query" === s) return i({}, e, {[s]: r});
                    if (!r.hasOwnProperty(t[s])) {
                        if (!r.hasOwnProperty("id")) throw new Error(`Ziggy error: object passed as '${s}' parameter is missing route model binding key '${t[s]}'.`);
                        t[s] = "id"
                    }
                    return i({}, e, {[s]: r[t[s]]})
                }, {})
            }

            h(e) {
                var t;
                let s = this.l.pathname.replace(this.t.url.replace(/^\w*:\/\/[^/]+/, ""), "").replace(/^\/+/, "");
                const r = (e, t = "", s) => {
                    const [r, n] = [e, t].map(e => e.split(s));
                    return n.reduce((e, t, s) => /^{[^}?]+\??}$/.test(t) && r[s] ? i({}, e, {[t.replace(/^{|\??}$/g, "")]: r[s]}) : e, {})
                };
                return i({}, r(this.l.host, e.domain, "."), r(s, e.uri, "/"), function (e, t) {
                    var s = k;
                    if ("" === e || null == e) return s.plainObjects ? Object.create(null) : {};
                    for (var r = "string" == typeof e ? function (e, t) {
                        var s, r = {},
                            n = (t.ignoreQueryPrefix ? e.replace(/^\?/, "") : e).split(t.delimiter, 1 / 0 === t.parameterLimit ? void 0 : t.parameterLimit),
                            i = -1, a = t.charset;
                        if (t.charsetSentinel) for (s = 0; s < n.length; ++s) 0 === n[s].indexOf("utf8=") && ("utf8=%E2%9C%93" === n[s] ? a = "utf-8" : "utf8=%26%2310003%3B" === n[s] && (a = "iso-8859-1"), i = s, s = n.length);
                        for (s = 0; s < n.length; ++s) if (s !== i) {
                            var o, l, c = n[s], d = c.indexOf("]="), p = -1 === d ? c.indexOf("=") : d + 1;
                            -1 === p ? (o = t.decoder(c, k.decoder, a, "key"), l = t.strictNullHandling ? null : "") : (o = t.decoder(c.slice(0, p), k.decoder, a, "key"), l = u.maybeMap(S(c.slice(p + 1), t), (function (e) {
                                return t.decoder(e, k.decoder, a, "value")
                            }))), l && t.interpretNumericEntities && "iso-8859-1" === a && (l = N(l)), c.indexOf("[]=") > -1 && (l = P(l) ? [l] : l), r[o] = E.call(r, o) ? u.combine(r[o], l) : l
                        }
                        return r
                    }(e, s) : e, n = s.plainObjects ? Object.create(null) : {}, i = Object.keys(r), a = 0; a < i.length; ++a) {
                        var o = i[a], l = T(o, r[o], s, "string" == typeof e);
                        n = u.merge(n, l, s)
                    }
                    return u.compact(n)
                }(null === (t = this.l.search) || void 0 === t ? void 0 : t.replace(/^\?/, "")))
            }

            valueOf() {
                return this.toString()
            }

            check(e) {
                return this.has(e)
            }
        }

        var M = function (e, t, s, r) {
            const n = new x(e, t, s, r);
            return e ? n.toString() : n
        };
        const R = Object(r.a)();

        function D(e, t, s) {
            return M(e, null != t ? t : {}, s, n).toString()
        }

        n.port = +R.port || null, n.url = R.origin
    }, "1BiD": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("0h6b"), n = s("/G9H"), i = s("tX/w");

        function a({group: e, modifiers: t}) {
            if (null == e) return null;
            let s, a = e.name;
            if (null != e.playmodes && e.playmodes.length > 0) {
                s = n.createElement("div", {className: "user-group-badge__modes"}, e.playmodes.map(e => n.createElement("i", {
                    key: e,
                    className: `fal fa-extra-mode-${e}`
                }))), a += ` (${e.playmodes.map(e => osu.trans(`beatmaps.mode.${e}`)).join(", ")})`
            }
            const o = {
                children: s,
                className: Object(i.a)("user-group-badge", {probationary: e.is_probationary}, e.identifier, t),
                "data-label": e.short_name,
                style: osu.groupColour(e),
                title: a
            };
            return e.has_listing ? n.createElement("a", Object.assign({}, o, {href: Object(r.a)("groups.show", {group: e.id})})) : n.createElement("div", Object.assign({}, o))
        }
    }, "205K": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("mkgZ");

        class n extends r.a {
            constructor(e) {
                super(), Object.defineProperty(this, "json", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, "2Kkx": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            const r = "js-mobile-toggle--active";

            class n {
                constructor() {
                    Object.defineProperty(this, "toggle", {
                        enumerable: !0, configurable: !0, writable: !0, value: e => {
                            const t = e.currentTarget;
                            if (!(t instanceof HTMLElement)) return;
                            const s = t.dataset.mobileToggleTarget;
                            if (null == s) return;
                            const n = document.querySelector(`.js-mobile-toggle[data-mobile-toggle-id=${s}]`);
                            if (!(n instanceof HTMLElement)) return;
                            const i = !t.classList.contains(r);
                            t.classList.toggle(r, i), n.classList.toggle("hidden-xs", !i)
                        }
                    }), e(document).on("click", ".js-mobile-toggle", this.toggle)
                }
            }
        }).call(this, s("5wds"))
    }, "2etm": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        })), s.d(t, "b", (function () {
            return n
        })), s.d(t, "c", (function () {
            return i
        }));
        const r = {
            announcement: ["fas fa-bullhorn"],
            beatmap_owner_change: ["fas fa-drafting-compass", "fas fa-user-friends"],
            beatmapset_discussion: ["fas fa-drafting-compass", "fas fa-comment"],
            beatmapset_problem: ["fas fa-drafting-compass", "fas fa-exclamation-circle"],
            beatmapset_state: ["fas fa-drafting-compass"],
            channel: ["fas fa-comments"],
            comment: ["fas fa-comment"],
            forum_topic_reply: ["fas fa-comment-medical"],
            legacy_pm: ["fas fa-envelope"],
            user_achievement_unlock: ["fas fa-medal"],
            user_beatmapset_new: ["fas fa-music"]
        }, n = {
            beatmap_owner_change: ["fas fa-drafting-compass", "fas fa-user-friends"],
            beatmapset_discussion_lock: ["fas fa-drafting-compass", "fas fa-lock"],
            beatmapset_discussion_post_new: ["fas fa-drafting-compass", "fas fa-comment-medical"],
            beatmapset_discussion_qualified_problem: ["fas fa-drafting-compass", "fas fa-exclamation-circle"],
            beatmapset_discussion_unlock: ["fas fa-drafting-compass", "fas fa-unlock"],
            beatmapset_disqualify: ["fas fa-drafting-compass", "far fa-times-circle"],
            beatmapset_love: ["fas fa-drafting-compass", "fas fa-heart"],
            beatmapset_nominate: ["fas fa-drafting-compass", "fas fa-vote-yea"],
            beatmapset_qualify: ["fas fa-drafting-compass", "fas fa-check"],
            beatmapset_rank: ["fas fa-drafting-compass", "fas fa-check-double"],
            beatmapset_remove_from_loved: ["fas fa-drafting-compass", "fas fa-heart-broken"],
            beatmapset_reset_nominations: ["fas fa-drafting-compass", "fas fa-undo"],
            channel_announcement: ["fas fa-bullhorn"],
            channel_message: ["fas fa-comments"],
            comment_new: ["fas fa-comment"],
            comment_reply: ["fas fa-reply"],
            forum_topic_reply: ["fas fa-comment-medical"],
            legacy_pm: ["fas fa-envelope"],
            user_achievement_unlock: ["fas fa-trophy"],
            user_beatmapset_new: ["fas fa-music"],
            user_beatmapset_revive: ["fas fa-drum"]
        }, i = {
            beatmap_owner_change: ["fas fa-user-edit"],
            beatmapset_discussion_lock: ["fas fa-lock"],
            beatmapset_discussion_post_new: ["fas fa-comment-medical"],
            beatmapset_discussion_qualified_problem: ["fas fa-exclamation-circle"],
            beatmapset_discussion_unlock: ["fas fa-unlock"],
            beatmapset_disqualify: ["far fa-times-circle"],
            beatmapset_love: ["fas fa-heart"],
            beatmapset_nominate: ["fas fa-vote-yea"],
            beatmapset_qualify: ["fas fa-check"],
            beatmapset_rank: ["fas fa-check-double"],
            beatmapset_remove_from_loved: ["fas fa-heart-broken"],
            beatmapset_reset_nominations: ["fas fa-undo"],
            channel_announcement: ["fas fa-bullhorn"],
            channel_message: ["fas fa-comments"],
            comment_new: ["fas fa-comment"],
            comment_reply: ["fas fa-reply"],
            forum_topic_reply: ["fas fa-comment-medical"],
            legacy_pm: ["fas fa-envelope"],
            user_beatmapset_new: ["fas fa-music"],
            user_beatmapset_revive: ["fas fa-drum"]
        }
    }, "2hxc": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("mkgZ");

        class n extends r.a {
            constructor(e) {
                super(), Object.defineProperty(this, "json", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, "2qTP": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return d
            }));
            var r, n = s("HUtF"), i = s("ceX/"), a = s("Hs9Z"), o = s("/G9H"), l = s("I8Ok"), c = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, u = {}.hasOwnProperty;
            r = "report-form";
            var d = function (t) {
                function s(t) {
                    this.sendReport = c(this.sendReport, this), this.renderFormContent = c(this.renderFormContent, this), this.renderForm = c(this.renderForm, this), this.render = c(this.render, this), this.handleReasonChange = c(this.handleReasonChange, this), this.handleCommentsChange = c(this.handleCommentsChange, this), s.__super__.constructor.call(this, t), this.options = [{
                        id: "Cheating",
                        text: osu.trans("users.report.options.cheating")
                    }, {
                        id: "MultipleAccounts",
                        text: osu.trans("users.report.options.multiple_accounts")
                    }, {id: "Insults", text: osu.trans("users.report.options.insults")}, {
                        id: "Spam",
                        text: osu.trans("users.report.options.spam")
                    }, {
                        id: "UnwantedContent",
                        text: osu.trans("users.report.options.unwanted_content")
                    }, {id: "Nonsense", text: osu.trans("users.report.options.nonsense")}, {
                        id: "Other",
                        text: osu.trans("users.report.options.other")
                    }], null != t.visibleOptions && (this.options = e.intersectionWith(this.options, t.visibleOptions, (function (e, t) {
                        return e.id === t
                    }))), this.state = {comments: "", selectedReason: this.options[0]}
                }

                return function (e, t) {
                    for (var s in t) u.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.handleCommentsChange = function (e) {
                    return this.setState({comments: e.target.value})
                }, s.prototype.handleReasonChange = function (e) {
                    return this.setState({selectedReason: e})
                }, s.prototype.render = function () {
                    return this.props.visible ? this.renderForm() : null
                }, s.prototype.renderForm = function () {
                    var e;
                    return e = this.props.completed ? osu.trans("users.report.thanks") : this.props.title, Object(o.createElement)(n.a, {
                        onClose: this.props.onClose,
                        visible: this.props.visible
                    }, Object(l.div)({className: r}, Object(l.div)({className: r + "__header"}, Object(l.div)({className: r + "__row " + r + "__row--exclamation"}, Object(l.i)({className: "fas fa-exclamation-triangle"})), Object(l.div)({
                        className: r + "__row " + r + "__row--title",
                        dangerouslySetInnerHTML: {__html: "<span>" + e + "</span>"}
                    })), this.props.completed ? void 0 : this.renderFormContent()))
                }, s.prototype.renderFormContent = function () {
                    return Object(l.div)(null, Object(a.isEmpty)(this.options) ? void 0 : [Object(l.div)({
                        key: "label",
                        className: r + "__row"
                    }, osu.trans("users.report.reason")), Object(l.div)({
                        key: "options",
                        className: r + "__row"
                    }, Object(o.createElement)(i.a, {
                        blackout: !1,
                        bn: r + "-select-options",
                        onChange: this.handleReasonChange,
                        options: this.options,
                        selected: this.state.selectedReason
                    }))], Object(l.div)({className: r + "__row"}, osu.trans("users.report.comments")), Object(l.div)({className: r + "__row"}, Object(l.textarea)({
                        className: r + "__textarea",
                        onChange: this.handleCommentsChange,
                        placeholder: osu.trans("users.report.placeholder"),
                        value: this.state.comments
                    })), Object(l.div)({className: r + "__row " + r + "__row--buttons"}, [Object(l.button)({
                        className: r + "__button " + r + "__button--report",
                        disabled: this.props.disabled || 0 === this.state.comments.length,
                        key: "report",
                        type: "button",
                        onClick: this.sendReport
                    }, osu.trans("users.report.actions.send")), Object(l.button)({
                        className: r + "__button",
                        disabled: this.props.disabled,
                        key: "cancel",
                        type: "button",
                        onClick: this.props.onClose
                    }, osu.trans("users.report.actions.cancel"))]))
                }, s.prototype.sendReport = function (e) {
                    var t, s, r;
                    return s = {
                        reason: null != (r = this.state.selectedReason) ? r.id : void 0,
                        comments: this.state.comments
                    }, "function" == typeof (t = this.props).onSubmit ? t.onSubmit(s) : void 0
                }, s
            }(o.PureComponent)
        }).call(this, s("Hs9Z"))
    }, "3J26": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("7EfK");

        class n {
            constructor() {
                Object.defineProperty(this, "observer", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "formatElem", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        "1" !== e.dataset.localtime && (e.dataset.localtime = "1", e.classList.add("js-tooltip-time"), e.title = e.dateTime, e.innerText = r(e.dateTime).format("LLL"))
                    }
                }), Object.defineProperty(this, "formatElems", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => (null == e && (e = this.getElems(document.body)), e.map(this.formatElem))
                }), Object.defineProperty(this, "getElems", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => e instanceof HTMLElement ? e instanceof HTMLTimeElement && e.classList.contains("js-localtime") ? [e] : [...e.querySelectorAll("time.js-localtime")] : []
                }), Object.defineProperty(this, "mutationHandler", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        const t = [];
                        e.forEach(e => {
                            e.addedNodes.forEach(e => {
                                t.push(...this.getElems(e))
                            })
                        }), this.formatElems(t)
                    }
                }), this.observer = new MutationObserver(this.mutationHandler), this.observer.observe(document, {
                    childList: !0,
                    subtree: !0
                })
            }
        }
    }, "3NvE": function (e, t, s) {
        "use strict";

        function r(e) {
        }

        s.d(t, "a", (function () {
            return r
        }))
    }, "3Wjd": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return r
            }));

            class r {
                constructor() {
                    Object.defineProperty(this, "loaded", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Set
                    }), Object.defineProperty(this, "loading", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "abortLoading", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            for (const e of this.loading.values()) e.abort()
                        }
                    }), Object.defineProperty(this, "forget", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            var t;
                            this.loaded.delete(e), null === (t = this.loading.get(e)) || void 0 === t || t.abort()
                        }
                    }), e(document).on("turbolinks:before-cache", this.abortLoading)
                }

                load(t) {
                    if (this.loaded.has(t) || this.loading.has(t)) return this.loading.get(t);
                    const s = e.ajax(t, {cache: !0, dataType: "script"});
                    return this.loading.set(t, s), s.done(() => this.loaded.add(t)).always(() => this.loading.delete(t)), s
                }

                loadSync(t) {
                    this.loaded.has(t) || (this.loading.has(t) && this.forget(t), e.ajax(t, {
                        async: !1,
                        cache: !0,
                        dataType: "script"
                    }), this.loaded.add(t))
                }
            }
        }).call(this, s("5wds"))
    }, "3XPW": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        function i(e) {
            const t = Object(n.a)("page-mode-link", "profile-page", {"is-active": e.page === e.currentPage}),
                s = osu.trans(`users.show.extra.${e.page}.title`);
            return r.createElement("span", {className: t}, r.createElement("span", {
                className: "fake-bold",
                "data-content": s
            }, s))
        }
    }, "3Zv4": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("y2EG"), n = s("71br"), i = s("f4vq");

        class a extends r.a {
            constructor() {
                super(-1, null), Object.defineProperty(this, "details", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: Object(n.a)()
                }), Object.defineProperty(this, "name", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: "legacy_pm"
                }), Object.defineProperty(this, "objectId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: -1
                }), this.isRead = !1
            }

            get count() {
                var e, t;
                return null !== (t = null === (e = i.a.currentUser) || void 0 === e ? void 0 : e.unread_pm_count) && void 0 !== t ? t : 0
            }
        }
    }, "3h35": function (e, t, s) {
        "use strict";
        var r = s("DiTM"), n = s("/HbY"), i = s("c1EF"), a = s("cxU/"), o = s("0h6b"), l = s("Hs9Z"), c = s("lv9K"),
            u = s("KUml"), d = s("f4vq"), p = s("/G9H"), m = s("tX/w"), h = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

        function b() {
        }

        let f = class extends p.Component {
            constructor(e) {
                super(e), Object.defineProperty(this, "onCoverExpandedToggle", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        d.a.userPreferences.set("profile_cover_expanded", !this.showCover)
                    }
                }), Object(c.p)(this)
            }

            get showCover() {
                return d.a.userPreferences.get("profile_cover_expanded")
            }

            render() {
                var e, t;
                return p.createElement("div", {className: Object(m.a)("profile-info", {cover: this.showCover})}, this.showCover && p.createElement("div", {
                    className: "profile-info__bg",
                    style: {backgroundImage: osu.urlPresence(this.props.coverUrl)}
                }, this.props.isUpdatingCover && p.createElement("div", {className: "profile-info__spinner"}, p.createElement(n.a, null)), this.props.editor), p.createElement("div", {className: "profile-info__details"}, this.props.user.id === (null === (e = d.a.currentUser) || void 0 === e ? void 0 : e.id) ? p.createElement("a", {
                    className: "profile-info__avatar",
                    href: `${Object(o.a)("account.edit")}#avatar`,
                    title: osu.trans("users.show.change_avatar")
                }, this.renderAvatar()) : p.createElement("div", {className: "profile-info__avatar"}, this.renderAvatar()), p.createElement("div", {className: "profile-info__info"}, p.createElement("h1", {className: "profile-info__name"}, p.createElement("span", {className: "u-ellipsis-pre-overflow"}, this.props.user.username), p.createElement("div", {className: "profile-info__previous-usernames"}, this.renderPreviousUsernames()), p.createElement("div", {className: "profile-info__icons profile-info__icons--name-inline"}, this.renderIcons())), this.renderTitle(), p.createElement("div", {className: "profile-info__flags"}, null != (null === (t = this.props.user.country) || void 0 === t ? void 0 : t.code) && p.createElement("a", {
                    className: "profile-info__flag",
                    href: Object(o.a)("rankings", {
                        country: this.props.user.country.code,
                        mode: this.props.currentMode,
                        type: "performance"
                    })
                }, p.createElement("span", {className: "profile-info__flag-flag"}, p.createElement(r.a, {country: this.props.user.country})), p.createElement("span", {className: "profile-info__flag-text"}, this.props.user.country.name)), p.createElement("div", {className: "profile-info__icons profile-info__icons--flag-inline"}, this.renderIcons()))), p.createElement("div", {className: "profile-info__cover-toggle"}, p.createElement("button", {
                    className: "btn-circle btn-circle--page-toggle",
                    onClick: this.onCoverExpandedToggle,
                    title: osu.trans(`users.show.cover.to_${this.showCover ? "0" : "1"}`),
                    type: "button"
                }, p.createElement("span", {className: this.showCover ? "fas fa-chevron-up" : "fas fa-chevron-down"})))))
            }

            renderAvatar() {
                return p.createElement(i.a, {modifiers: "full", user: this.props.user})
            }

            renderIcons() {
                var e;
                return p.createElement(p.Fragment, null, this.props.user.is_supporter && p.createElement("span", {
                    className: "profile-info__icon profile-info__icon--supporter",
                    title: osu.trans("users.show.is_supporter")
                }, Object(l.times)(null !== (e = this.props.user.support_level) && void 0 !== e ? e : 0, e => p.createElement("span", {
                    key: e,
                    className: "fas fa-heart"
                }))), p.createElement(a.a, {
                    groups: this.props.user.groups,
                    modifiers: "profile-page",
                    wrapper: "profile-info__icon"
                }))
            }

            renderPreviousUsernames() {
                if (null == this.props.user.previous_usernames || 0 === this.props.user.previous_usernames.length) return null;
                const e = this.props.user.previous_usernames.join(", ");
                return p.createElement("div", {className: "profile-previous-usernames"}, p.createElement("a", {
                    className: "profile-previous-usernames__icon profile-previous-usernames__icon--with-title",
                    onClick: b,
                    title: `${osu.trans("users.show.previous_usernames")}: ${e}`
                }, p.createElement("span", {className: "fas fa-address-card"})), p.createElement("div", {className: "profile-previous-usernames__icon profile-previous-usernames__icon--plain"}, p.createElement("span", {className: "fas fa-address-card"})), p.createElement("div", {className: "profile-previous-usernames__content"}, p.createElement("div", {className: "profile-previous-usernames__title"}, osu.trans("users.show.previous_usernames")), p.createElement("div", {className: "profile-previous-usernames__names"}, e)))
            }

            renderTitle() {
                var e;
                if (null == this.props.user.title) return null;
                const t = {
                    children: this.props.user.title,
                    className: "profile-info__title",
                    style: {color: null !== (e = this.props.user.profile_colour) && void 0 !== e ? e : void 0}
                };
                return null != this.props.user.title_url ? p.createElement("a", Object.assign({href: this.props.user.title_url}, t)) : p.createElement("span", Object.assign({}, t))
            }
        };
        h([c.h], f.prototype, "showCover", null), f = h([u.b], f), t.a = f
    }, "4PJi": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("0h6b"), n = s("f4vq"), i = s("/DQ7"), a = s("phBA");
            const o = e => {
                var t;
                return 401 === e.status && "verify" === (null === (t = e.responseJSON) || void 0 === t ? void 0 : t.authentication)
            };

            class l {
                constructor() {
                    Object.defineProperty(this, "callback", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "delayShow", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "delayShowCallback", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "modal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "request", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "showOnError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => !!o(e) && (this.show(e.responseJSON.box, t), !0)
                    }), Object.defineProperty(this, "autoSubmit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var t;
                            const s = this.inputBox;
                            if (null == s) return;
                            const n = s.value.replace(/\s/g, ""), i = s.dataset.lastKey,
                                a = parseInt(null !== (t = s.dataset.verificationKeyLength) && void 0 !== t ? t : "", 10);
                            0 === n.length && this.setMessage(), a === n.length && n !== i && (s.dataset.lastKey = n, this.prepareForRequest("verifying"), this.request = e.post(Object(r.a)("account.verify"), {verification_key: n}).done(this.success).fail(this.error))
                        }
                    }), Object.defineProperty(this, "error", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.setMessage(osu.xhrErrorMessage(e))
                        }
                    }), Object.defineProperty(this, "float", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t, s) => {
                            e ? (t.classList.add("js-user-verification--center"), t.style.paddingTop = "") : (t.classList.remove("js-user-verification--center"), t.style.paddingTop = `${null != s ? s : 0}px`)
                        }
                    }), Object.defineProperty(this, "isActive", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e;
                            return null === (e = this.modal) || void 0 === e ? void 0 : e.classList.contains("js-user-verification--active")
                        }
                    }), Object.defineProperty(this, "onError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => this.showOnError(t, Object(a.c)(e.target))
                    }), Object.defineProperty(this, "prepareForRequest", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            var t;
                            null === (t = this.request) || void 0 === t || t.abort(), this.setMessage(osu.trans(`user_verification.box.${e}`), !0)
                        }
                    }), Object.defineProperty(this, "reissue", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            t.preventDefault(), this.prepareForRequest("issuing"), this.request = e.post(Object(r.a)("account.reissue-code")).done(e => {
                                this.setMessage(e.message)
                            }).fail(this.error)
                        }
                    }), Object.defineProperty(this, "reposition", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e, t;
                            if (this.isActive() && null != this.modal) if (n.a.windowSize.isMobile) this.float(!0, this.modal); else {
                                const s = null !== (t = null === (e = this.reference) || void 0 === e ? void 0 : e.getBoundingClientRect().bottom) && void 0 !== t ? t : 0;
                                this.float(s < 0, this.modal, s)
                            }
                        }
                    }), Object.defineProperty(this, "setDelayShow", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.delayShow = !0
                        }
                    }), Object.defineProperty(this, "setMessage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t = !1) => {
                            const s = this.message;
                            if (null == s) return;
                            if (null == e || 0 === e.length) return void Object(i.b)(s);
                            const r = this.messageText, n = this.messageSpinner;
                            null != r && null != n && (r.textContent = e, Object(i.c)(n, t), Object(i.a)(s))
                        }
                    }), Object.defineProperty(this, "setModal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = document.querySelector(".js-user-verification");
                            this.modal = e instanceof HTMLElement ? e : void 0
                        }
                    }), Object.defineProperty(this, "show", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            this.delayShow ? this.delayShowCallback = () => this.show(t, s) : (this.callback = s, null != t && e(".js-user-verification--box").html(t), this.$modal().modal({
                                backdrop: "static",
                                keyboard: !1,
                                show: !0
                            }).addClass("js-user-verification--active"), this.reposition())
                        }
                    }), Object.defineProperty(this, "showOnLoad", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.delayShow = !1, null != this.delayShowCallback ? (this.delayShowCallback(), this.delayShowCallback = void 0) : this.isVerificationPage() && this.show()
                        }
                    }), Object.defineProperty(this, "success", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (!this.isActive() || null == this.modal) return;
                            const e = this.inputBox;
                            if (null == e) return;
                            this.$modal().modal("hide"), this.modal.classList.remove("js-user-verification--active");
                            const t = this.callback;
                            if (this.callback = void 0, this.setMessage(), e.value = "", e.dataset.lastKey = "", this.isVerificationPage()) return osu.reloadPage();
                            null == t || t()
                        }
                    }), e(document).on("ajax:error", this.onError).on("turbolinks:load", this.setModal).on("turbolinks:load", this.showOnLoad).on("turbolinks:visit", this.setDelayShow).on("input", ".js-user-verification--input", this.autoSubmit).on("click", ".js-user-verification--reissue", this.reissue), e.subscribe("user-verification:success", this.success), e(window).on("resize scroll", this.reposition)
                }

                get inputBox() {
                    return document.querySelector(".js-user-verification--input")
                }

                get message() {
                    return document.querySelector(".js-user-verification--message")
                }

                get messageSpinner() {
                    return document.querySelector(".js-user-verification--message-spinner")
                }

                get messageText() {
                    return document.querySelector(".js-user-verification--message-text")
                }

                get reference() {
                    return document.querySelector(".js-user-verification--reference")
                }

                $modal() {
                    return e(".js-user-verification")
                }

                isVerificationPage() {
                    return null != document.querySelector(".js-user-verification--on-load")
                }
            }
        }).call(this, s("5wds"))
    }, "4mNh": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("0h6b"), n = s("/G9H"), i = s("tX/w");

        function a(e) {
            return n.createElement("div", {className: "page-tabs page-tabs--follows"}, ["comment", "forum_topic", "mapping", "modding"].map(t => n.createElement("a", {
                key: t,
                className: Object(i.a)("page-tabs__tab", {active: t === e.currentSubtype}),
                href: Object(r.a)("follows.index", {subtype: t})
            }, osu.trans(`follows.${t}.title`))))
        }
    }, "4tBt": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return d
        }));
        var r = s("Kexm"), n = s("0h6b"), i = s("KUml"), a = s("f4vq"), o = s("/G9H"), l = s("u/q5"), c = s("FpSo"),
            u = s("B5xN");
        let d = class extends o.Component {
            render() {
                const {score: e} = this.props;
                return o.createElement(c.a, null, t => {
                    var s, i;
                    return o.createElement("div", {className: "simple-menu"}, a.a.scorePins.canBePinned(e) && o.createElement(r.a, {
                        className: "simple-menu__item",
                        onUpdate: t,
                        score: e
                    }), Object(l.d)(e) && o.createElement("a", {
                        className: "simple-menu__item",
                        href: Object(n.a)("scores.show", {mode: e.mode, score: e.best_id})
                    }, osu.trans("users.show.extra.top_ranks.view_details")), Object(l.c)(e) && o.createElement("a", {
                        className: "simple-menu__item js-login-required--click",
                        "data-turbolinks": !1,
                        href: Object(n.a)("scores.download", {mode: e.mode, score: e.best_id}),
                        onClick: t
                    }, osu.trans("users.show.extra.top_ranks.download_replay")), Object(l.a)(e) && o.createElement(u.a, {
                        baseKey: "scores",
                        className: "simple-menu__item",
                        reportableId: null !== (i = null === (s = e.best_id) || void 0 === s ? void 0 : s.toString()) && void 0 !== i ? i : "",
                        reportableType: `score_best_${e.mode}`,
                        user: e.user
                    }))
                })
            }
        };
        d = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([i.b], d)
    }, "55pz": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");

        function n(e) {
            const t = Object.keys(e.mappings);
            if (0 === t.length) return r.createElement(r.Fragment, null, e.pattern);
            const s = new RegExp(`(:${t.join("|:")})`), n = e.pattern.split(s);
            return r.createElement(r.Fragment, null, n.map(t => {
                const s = ":" === t[0] ? t.slice(1) : null, n = null == s || null == e.mappings[s] ? t : e.mappings[s];
                return "string" == typeof n ? n : r.createElement(r.Fragment, {key: t}, n)
            }))
        }
    }, "5AHq": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = r.createContext([])
    }, "5b8Q": function (e, t, s) {
        "use strict";
        (function (e) {
            function r() {
                return window.setTimeout(n, 0)
            }

            function n() {
                e.publish("osu:page:change")
            }

            s.d(t, "a", (function () {
                return r
            })), s.d(t, "b", (function () {
                return n
            }))
        }).call(this, s("5wds"))
    }, "5e07": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        const r = e => e.toString().padStart(2, "0"), n = {}, i = (e, t) => {
            var s;
            const i = Math.floor(e);
            if (n.time !== i || n.format !== t) {
                n.format = t, n.time = i;
                const e = i % 60, s = Math.floor(i / 60);
                if ("minute_minimal" === t) n.formatted = `'${s}:${r(e)}'`; else if ("minute" === t) n.formatted = `'${r(s)}:${r(e)}'`; else {
                    const i = s % 60, a = Math.floor(s / 60);
                    n.formatted = "hour_minimal" === t ? `'${a}:${r(i)}:${r(e)}'` : `'${r(a)}:${r(i)}:${r(e)}'`
                }
            }
            return null !== (s = n.formatted) && void 0 !== s ? s : ""
        }
    }, "5eFc": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        class i extends r.Component {
            render() {
                const e = this.props.comments.filter(e => null != e.deletedAt).length;
                return 0 === e ? null : r.createElement("div", {className: Object(n.a)("deleted-comments-count", this.props.modifiers)}, r.createElement("span", {className: "deleted-comments-count__icon"}, r.createElement("span", {className: "far fa-trash-alt"})), osu.transChoice("comments.deleted_count", e))
            }
        }
    }, "5evE": function (e, t, s) {
        "use strict";

        function r(e) {
            return null == e.objectId || null == e.name || null == e.category ? null : "user_achievement_unlock" === e.name ? `achievement:${e.id}` : `${e.category}:${e.objectId}`
        }

        function n(e) {
            var t;
            return null !== (t = i[e]) && void 0 !== t ? t : e
        }

        s.d(t, "b", (function () {
            return r
        })), s.d(t, "a", (function () {
            return n
        }));
        const i = {
            beatmap_owner_change: "beatmap_owner_change",
            beatmapset_discussion_lock: "beatmapset_discussion",
            beatmapset_discussion_post_new: "beatmapset_discussion",
            beatmapset_discussion_qualified_problem: "beatmapset_problem",
            beatmapset_discussion_review_new: "beatmapset_discussion",
            beatmapset_discussion_unlock: "beatmapset_discussion",
            beatmapset_disqualify: "beatmapset_state",
            beatmapset_love: "beatmapset_state",
            beatmapset_nominate: "beatmapset_state",
            beatmapset_qualify: "beatmapset_state",
            beatmapset_rank: "beatmapset_state",
            beatmapset_remove_from_loved: "beatmapset_state",
            beatmapset_reset_nominations: "beatmapset_state",
            channel_announcement: "announcement",
            channel_message: "channel",
            comment_new: "comment",
            comment_reply: "comment",
            forum_topic_reply: "forum_topic_reply",
            legacy_pm: "legacy_pm",
            user_achievement_unlock: "user_achievement_unlock",
            user_beatmapset_new: "user_beatmapset_new",
            user_beatmapset_revive: "user_beatmapset_new"
        }
    }, "5hCN": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            s("/G9H");
            var r, n, i = s("I8Ok");
            r = "beatmap-basic-stats", n = function (t) {
                var s, r, n;
                return n = t % 60, r = Math.floor(t / 60) % 60, (s = Math.floor(t / 3600)) > 0 ? s + ":" + e.padStart(r, 2, 0) + ":" + e.padStart(n, 2, 0) : r + ":" + e.padStart(n, 2, 0)
            };
            var a = function (e) {
                var t, s, a;
                return t = e.beatmap, Object(i.div)({className: r}, function () {
                    var e, o, l, c;
                    for (c = [], e = 0, o = (l = ["total_length", "bpm", "count_circles", "count_sliders"]).length; e < o; e++) a = t[s = l[e]], a = "bpm" === s ? a > 1e3 ? "âˆž" : osu.formatNumber(a) : "total_length" === s ? n(a) : osu.formatNumber(a), c.push(Object(i.div)({
                        className: r + "__entry",
                        key: s,
                        title: osu.trans("beatmapsets.show.stats." + s, "total_length" === s ? {hit_length: n(t.hit_length)} : void 0)
                    }, Object(i.div)({
                        className: r + "__entry-icon",
                        style: {backgroundImage: "url(/images/layout/beatmapset-page/" + s + ".svg)"}
                    }), Object(i.span)(null, a)));
                    return c
                }())
            }
        }).call(this, s("Hs9Z"))
    }, "6QPw": function (e, t, s) {
        "use strict";
        var r = s("KUml"), n = s("3Zv4"), i = s("2etm"), a = s("mjdM"), o = s("/G9H");
        let l = class extends o.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "handleMarkAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                })
            }

            render() {
                const e = new n.a;
                return 0 === e.count ? null : o.createElement(a.a, {
                    icons: i.b.legacy_pm,
                    item: e,
                    message: osu.transChoice("notifications.item.legacy_pm.legacy_pm.legacy_pm", e.count),
                    modifiers: ["one"],
                    url: "/forum/ucp.php?i=pm&folder=inbox",
                    withCategory: !0,
                    withCoverImage: !0
                })
            }
        };
        l = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], l), t.a = l
    }, "6WEK": function (e, t, s) {
        "use strict";
        s.d(t, "b", (function () {
            return P
        })), s.d(t, "a", (function () {
            return k
        }));
        var r = s("7nlf"), n = s("VbpL"), i = s("0h6b"), a = s("Hs9Z"), o = s("f4vq"), l = s("/G9H"), c = s("tX/w"),
            u = s("DiTM"), d = s("PdfH"), p = s("FpSo"), m = s("B5xN"), h = s("/HbY"), b = s("55pz"), f = s("rMK6"),
            v = s("UBw1"), g = s("lv9K"), y = s("KUml"), w = s("oQBk"), _ = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };
        let O = class extends l.Component {
            constructor(e) {
                super(e), Object(g.p)(this)
            }

            get friendModifier() {
                var e;
                if (null == (null === (e = o.a.currentUser) || void 0 === e ? void 0 : e.friends)) return;
                const t = o.a.currentUser.friends.find(e => e.target_id === this.props.user.id);
                if (null != t) {
                    if (t.mutual) return "mutual";
                    if (!this.context.isFriendsPage) return "friend"
                }
            }

            render() {
                var e, t, s;
                const r = Object(c.a)("user-card-brick", this.props.modifiers, this.props.mode, this.friendModifier);
                return l.createElement("a", {
                    className: `js-usercard ${r}`,
                    "data-user-id": this.props.user.id,
                    href: Object(i.a)("users.show", {user: this.props.user.id})
                }, l.createElement("div", {
                    className: "user-card-brick__group-bar",
                    style: osu.groupColour(null === (e = this.props.user.groups) || void 0 === e ? void 0 : e[0]),
                    title: null === (s = null === (t = this.props.user.groups) || void 0 === t ? void 0 : t[0]) || void 0 === s ? void 0 : s.name
                }), l.createElement("div", {className: "user-card-brick__username u-ellipsis-overflow"}, this.props.user.username))
            }
        };
        Object.defineProperty(O, "contextType", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: w.a
        }), Object.defineProperty(O, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {mode: "brick"}
        }), _([g.h], O.prototype, "friendModifier", null);
        var j = O = _([y.b], O), E = s("cxU/");
        const P = ["card", "list", "brick"];

        class k extends l.PureComponent {
            constructor() {
                super(...arguments), Object.defineProperty(this, "state", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {avatarLoaded: !1, backgroundLoaded: !1}
                }), Object.defineProperty(this, "url", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "onAvatarLoad", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({avatarLoaded: !0})
                    }
                }), Object.defineProperty(this, "onBackgroundLoad", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({backgroundLoaded: !0})
                    }
                })
            }

            get canMessage() {
                var e, t;
                return !this.isSelf && null == a.find(null !== (t = null === (e = o.a.currentUser) || void 0 === e ? void 0 : e.blocks) && void 0 !== t ? t : [], {target_id: this.user.id})
            }

            get isOnline() {
                return this.user.is_online
            }

            get isSelf() {
                return null != o.a.currentUser && o.a.currentUser.id === this.user.id
            }

            get isUserLoaded() {
                return Number.isFinite(this.user.id) && this.user.id > 0
            }

            get isUserNotFound() {
                return -1 === this.user.id
            }

            get isUserVisible() {
                return this.isUserLoaded && !this.user.is_deleted
            }

            get user() {
                return this.props.user || k.userLoading
            }

            render() {
                if ("brick" === this.props.mode) return null == this.props.user ? null : l.createElement(j, Object.assign({}, this.props, {user: this.props.user}));
                const e = this.props.modifiers.slice();
                return e.push(this.props.activated ? "active" : "highlightable"), e.push(this.props.mode), this.url = this.isUserVisible ? Object(i.a)("users.show", {user: this.user.id}) : void 0, l.createElement("div", {className: Object(c.a)("user-card", e)}, this.renderBackground(), l.createElement("div", {className: "user-card__card"}, l.createElement("div", {className: "user-card__content user-card__content--details"}, l.createElement("div", {className: "user-card__user"}, this.renderAvatar()), l.createElement("div", {className: "user-card__details"}, this.renderIcons(), l.createElement("div", {className: "user-card__username-row"}, this.renderUsername(), l.createElement("div", {className: "user-card__group-badges"}, l.createElement(E.a, {
                    groups: this.user.groups,
                    short: !0,
                    wrapper: "user-card__group-badge"
                }))), this.renderListModeIcons())), this.renderStatusBar()))
            }

            renderAvatar() {
                const e = {loaded: this.state.avatarLoaded},
                    t = osu.present(this.user.avatar_url) && !this.isUserNotFound;
                return l.createElement("div", {className: "user-card__avatar-space"}, l.createElement("div", {className: Object(c.a)("user-card__avatar-spinner", e)}, t && l.createElement(h.a, {modifiers: e})), this.isUserLoaded && t && l.createElement("img", {
                    className: Object(c.a)("user-card__avatar", e),
                    onError: this.onAvatarLoad,
                    onLoad: this.onAvatarLoad,
                    src: this.user.avatar_url
                }))
            }

            renderBackground() {
                let e, t;
                const s = Object(c.a)("user-card__background-overlay", this.isOnline ? ["online"] : []);
                if (this.user.cover && this.user.cover.url) {
                    let t = "user-card__background";
                    this.state.backgroundLoaded || (t += " user-card__background--loading"), e = l.createElement(l.Fragment, null, l.createElement("img", {
                        className: t,
                        onLoad: this.onBackgroundLoad,
                        src: this.user.cover.url
                    }), l.createElement("div", {className: s}))
                } else e = l.createElement("div", {className: s});
                return t = this.isUserVisible ? l.createElement("a", {
                    className: "user-card__background-container",
                    href: this.url
                }, e) : e
            }

            renderIcons() {
                return this.isUserVisible ? l.createElement("div", {className: "user-card__icons user-card__icons--card"}, l.createElement("a", {
                    className: "user-card__icon user-card__icon--flag",
                    href: Object(i.a)("rankings", {country: this.user.country_code, mode: "osu", type: "performance"})
                }, l.createElement(u.a, {country: this.user.country})), "card" === this.props.mode && l.createElement(l.Fragment, null, this.user.is_supporter && l.createElement("a", {
                    className: "user-card__icon",
                    href: Object(i.a)("support-the-game")
                }, l.createElement(f.a, {modifiers: ["user-card"]})), l.createElement("div", {className: "user-card__icon"}, l.createElement(n.a, {
                    modifiers: "user-card",
                    userId: this.user.id
                })), !this.user.is_bot && l.createElement("div", {className: "user-card__icon"}, l.createElement(d.a, {
                    modifiers: ["user-card"],
                    userId: this.user.id
                })))) : null
            }

            renderListModeIcons() {
                return "list" === this.props.mode && this.isUserVisible ? l.createElement("div", {className: "user-card__icons"}, this.user.is_supporter && l.createElement("a", {
                    className: "user-card__icon",
                    href: Object(i.a)("support-the-game")
                }, l.createElement(f.a, {
                    level: this.user.support_level,
                    modifiers: ["user-list"]
                })), l.createElement("div", {className: "user-card__icon"}, l.createElement(n.a, {
                    modifiers: "user-list",
                    userId: this.user.id
                })), !this.user.is_bot && l.createElement("div", {className: "user-card__icon"}, l.createElement(d.a, {
                    modifiers: ["user-list"],
                    userId: this.user.id
                }))) : null
            }

            renderMenuButton() {
                if (this.isSelf) return null;
                return l.createElement("div", {className: "user-card__icon user-card__icon--menu"}, l.createElement(p.a, null, e => l.createElement("div", {className: "simple-menu"}, this.canMessage && l.createElement("a", {
                    className: "simple-menu__item js-login-required--click",
                    href: Object(i.a)("messages.users.show", {user: this.user.id}),
                    onClick: e
                }, l.createElement("span", {className: "fas fa-envelope"}), ` ${osu.trans("users.card.send_message")}`), l.createElement(r.a, {
                    modifiers: "inline",
                    onClick: e,
                    userId: this.user.id,
                    wrapperClass: "simple-menu__item"
                }), l.createElement(m.a, {
                    className: "simple-menu__item",
                    icon: !0,
                    onFormClose: e,
                    reportableId: this.user.id.toString(),
                    reportableType: "user",
                    user: this.user
                }))))
            }

            renderStatusBar() {
                if (!this.isUserVisible) return null;
                const e = this.isOnline ? osu.trans("users.status.online") : osu.trans("users.status.offline");
                return l.createElement("div", {className: "user-card__content user-card__content--status"}, l.createElement("div", {className: "user-card__status"}, this.renderStatusIcon(), l.createElement("div", {className: "user-card__status-messages"}, l.createElement("span", {className: "user-card__status-message user-card__status-message--sub u-ellipsis-overflow"}, !this.isOnline && null != this.user.last_visit && l.createElement(b.a, {
                    mappings: {
                        date: l.createElement(v.a, {
                            dateTime: this.user.last_visit,
                            relative: !0
                        })
                    }, pattern: osu.trans("users.show.lastvisit")
                })), l.createElement("span", {className: "user-card__status-message u-ellipsis-overflow"}, e))), l.createElement("div", {className: "user-card__icons user-card__icons--menu"}, this.renderMenuButton()))
            }

            renderStatusIcon() {
                return this.isUserVisible ? l.createElement("div", {className: "user-card__status-icon-container"}, l.createElement("div", {className: `user-card__status-icon user-card__status-icon--${this.isOnline ? "online" : "offline"}`})) : null
            }

            renderUsername() {
                const e = this.user.is_deleted ? osu.trans("users.deleted") : this.user.username;
                return null == this.url ? l.createElement("div", {className: "user-card__username u-ellipsis-pre-overflow"}, e) : l.createElement("a", {
                    className: "user-card__username u-ellipsis-pre-overflow",
                    href: this.url
                }, e)
            }
        }

        Object.defineProperty(k, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {activated: !1, mode: "card", modifiers: []}
        }), Object.defineProperty(k, "userLoading", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {
                avatar_url: "",
                country_code: "",
                cover: {custom_url: null, id: null, url: null},
                default_group: "",
                id: 0,
                is_active: !1,
                is_bot: !1,
                is_deleted: !1,
                is_online: !1,
                is_supporter: !1,
                last_visit: "",
                pm_friends_only: !0,
                profile_colour: "",
                username: osu.trans("users.card.loading")
            }
        })
    }, "6b0J": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = r.createContext({})
    }, "6se0": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("0h6b"), n = s("lv9K"), i = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

            class a {
                constructor() {
                    Object.defineProperty(this, "pins", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object(n.p)(this)
                }

                apiPin(t, s) {
                    const i = t.current_user_attributes.pin;
                    if (null == i) throw new Error("can't pin score without current user attributes");
                    return e.ajax(Object(r.a)("score-pins.store"), {
                        data: i,
                        dataType: "json",
                        method: s ? "POST" : "DELETE"
                    }).done(Object(n.f)(() => {
                        this.markPinned(t, s), e.publish("score:pin", [s, t])
                    }))
                }

                canBePinned(e) {
                    return null != e.current_user_attributes.pin
                }

                isPinned(e) {
                    const t = e.current_user_attributes.pin;
                    if (null == t) return !1;
                    const s = this.mapKey(t);
                    return this.pins.has(s) || Object(n.u)(() => {
                        this.pins.set(s, t.is_pinned)
                    }), this.pins.get(s)
                }

                markPinned(e, t) {
                    const s = e.current_user_attributes.pin;
                    if (null == s) return;
                    const r = this.mapKey(s);
                    if (null == r) return null;
                    this.pins.set(r, t)
                }

                mapKey(e) {
                    return `${e.score_type}:${e.score_id}`
                }
            }

            i([n.q], a.prototype, "pins", void 0), i([n.f], a.prototype, "markPinned", null)
        }).call(this, s("5wds"))
    }, "70Tv": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            var r = s("V/vD");

            class n {
                constructor() {
                    Object.defineProperty(this, "layzr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "observer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "observePage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (null != this.layzr) for (const t of e) for (const e of t.addedNodes) if (e instanceof HTMLElement && (null != e.dataset.normal || null != e.querySelector("[data-normal]"))) return void this.reinit()
                        }
                    }), this.observer = new MutationObserver(this.observePage), this.observer.observe(document, {
                        childList: !0,
                        subtree: !0
                    }), e(() => {
                        var e;
                        null !== (e = this.layzr) && void 0 !== e || (this.layzr = Object(r.a)()), this.reinit()
                    })
                }

                reinit() {
                    var e;
                    null === (e = this.layzr) || void 0 === e || e.update().check().handlers(!0)
                }
            }
        }).call(this, s("5wds"))
    }, "71br": function (e, t, s) {
        "use strict";

        function r() {
            return {coverUrl: null, title: ""}
        }

        s.d(t, "a", (function () {
            return r
        }))
    }, "74hk": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = Object(r.createContext)(void 0)
    }, "76+M": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("0h6b");
        const n = ["modding", "playlists", "realtime"], i = {
            modding: e => Object(r.a)("users.modding.index", {user: e}),
            playlists: e => Object(r.a)("users.multiplayer.index", {typeGroup: "playlists", user: e}),
            realtime: e => Object(r.a)("users.multiplayer.index", {typeGroup: "realtime", user: e}),
            show: e => Object(r.a)("users.show", {user: e})
        };

        function a(e, t, s) {
            return {active: t, title: osu.trans(`layout.header.users.${e}`), url: i[e](s)}
        }

        function o(e, t) {
            const s = [a("show", "show" === t, e.id)];
            return e.is_bot || n.forEach(r => {
                s.push(a(r, t === r, e.id))
            }), s
        }
    }, "7nlf": function (e, t, s) {
        "use strict";
        (function (e) {
            var r = s("0h6b"), n = s("lv9K"), i = s("KUml"), a = s("f4vq"), o = s("/G9H"), l = s("/jJF"), c = s("tX/w"),
                u = s("/HbY"), d = function (e, t, s, r) {
                    var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };
            const p = "textual-button";
            let m = class extends o.Component {
                constructor(t) {
                    super(t), Object.defineProperty(this, "loading", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "clicked", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e, t;
                            confirm(osu.trans("common.confirmation")) ? this.toggleBlock() : null === (t = (e = this.props).onClick) || void 0 === t || t.call(e)
                        }
                    }), Object.defineProperty(this, "toggleBlock", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.loading = !0, null == this.block ? this.xhr = e.ajax(Object(r.a)("blocks.store", {target: this.props.userId}), {type: "POST"}) : this.xhr = e.ajax(Object(r.a)("blocks.destroy", {block: this.props.userId}), {type: "DELETE"}), this.xhr.done(this.updateBlocks).fail(Object(l.c)(this.toggleBlock)).always(Object(n.f)(() => this.loading = !1))
                        }
                    }), Object.defineProperty(this, "updateBlocks", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            var s, r;
                            null != a.a.currentUser && (a.a.currentUser.blocks = t.filter(e => "block" === e.relation_type), a.a.currentUser.friends = t.filter(e => "friend" === e.relation_type), e.publish("user:update", a.a.currentUser)), null === (r = (s = this.props).onClick) || void 0 === r || r.call(s)
                        }
                    }), Object(n.p)(this)
                }

                get block() {
                    var e;
                    return null === (e = a.a.currentUser) || void 0 === e ? void 0 : e.blocks.find(e => e.target_id === this.props.userId)
                }

                get isVisible() {
                    return null != a.a.currentUser && Number.isFinite(this.props.userId) && this.props.userId !== a.a.currentUser.id
                }

                componentWillUnmount() {
                    var e;
                    null === (e = this.xhr) || void 0 === e || e.abort()
                }

                render() {
                    if (!this.isVisible) return null;
                    const e = Object(c.a)(p, this.props.modifiers, "block");
                    let t, s;
                    return null == this.props.wrapperClass ? s = e : (t = e, s = this.props.wrapperClass), o.createElement("button", {
                        className: s,
                        disabled: this.loading,
                        onClick: this.clicked,
                        type: "button"
                    }, o.createElement("span", {className: t}, this.loading ? o.createElement("span", {className: `${p}__icon fa-fw`}, o.createElement(u.a, null)) : o.createElement("span", {className: `${p}__icon fas fa-ban fa-fw`}), " ", null == this.block ? osu.trans("users.blocks.button.block") : osu.trans("users.blocks.button.unblock")))
                }
            };
            d([n.q], m.prototype, "loading", void 0), d([n.h], m.prototype, "block", null), d([n.h], m.prototype, "isVisible", null), d([n.f], m.prototype, "toggleBlock", void 0), d([n.f], m.prototype, "updateBlocks", void 0), m = d([i.b], m), t.a = m
        }).call(this, s("5wds"))
    }, "8Xmz": function (e, t, s) {
        "use strict";
        var r = s("/HbY"), n = s("KUml"), i = s("/G9H"), a = s("tX/w"), o = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let l = class extends i.Component {
            render() {
                return this.props.isDeleting ? i.createElement("div", {className: Object(a.a)("notification-action-button", this.props.modifiers)}, i.createElement("span", {className: "notification-action-button__text"}, this.props.text), i.createElement("div", {className: "notification-action-button__icon"}, i.createElement(r.a, null))) : i.createElement("button", {
                    className: Object(a.a)("notification-action-button", this.props.modifiers),
                    onClick: this.props.onDelete,
                    type: "button"
                }, i.createElement("span", {className: "notification-action-button__text"}, this.props.text), i.createElement("div", {className: "notification-action-button__icon"}, i.createElement("span", {className: "fas fa-trash"})))
            }
        };
        Object.defineProperty(l, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {modifiers: []}
        }), l = o([n.b], l), t.a = l
    }, "8fnM": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return v
            }));
            var r = s("7EfK"), n = s("f4vq"), i = s("/G9H"), a = s("tX/w"), o = s("is6n"), l = s("vZz4"), c = s("FR9d"),
                u = s("6WEK"), d = s("NfKI");
            const p = ["all", "online", "offline"], m = ["all", "osu", "taiko", "fruits", "mania"],
                h = ["last_visit", "rank", "username"];

            function b(e, t) {
                var s, r, n, i;
                return (null !== (r = null === (s = e.statistics) || void 0 === s ? void 0 : s.global_rank) && void 0 !== r ? r : Number.MAX_VALUE) - (null !== (i = null === (n = t.statistics) || void 0 === n ? void 0 : n.global_rank) && void 0 !== i ? i : Number.MAX_VALUE)
            }

            function f(e, t) {
                return e.username.localeCompare(t.username)
            }

            class v extends i.PureComponent {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "state", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {
                            filter: this.filterFromUrl,
                            playMode: this.playmodeFromUrl,
                            sortMode: this.sortFromUrl,
                            viewMode: this.viewFromUrl
                        }
                    }), Object.defineProperty(this, "handleSortChange", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = t.currentTarget.dataset.value, r = Object(l.i)(null, {sort: s});
                            e.controller.advanceHistory(r), this.setState({sortMode: s}, () => {
                                n.a.userPreferences.set("user_list_sort", this.state.sortMode)
                            })
                        }
                    }), Object.defineProperty(this, "onViewSelected", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = t.currentTarget.dataset.value, r = Object(l.i)(null, {view: s});
                            e.controller.advanceHistory(r), this.setState({viewMode: s}, () => {
                                n.a.userPreferences.set("user_list_view", this.state.viewMode)
                            })
                        }
                    }), Object.defineProperty(this, "optionSelected", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            t.preventDefault();
                            const s = t.currentTarget.dataset.key, r = Object(l.i)(null, {filter: s});
                            e.controller.advanceHistory(r), this.setState({filter: s}, () => {
                                n.a.userPreferences.set("user_list_filter", this.state.filter)
                            })
                        }
                    }), Object.defineProperty(this, "playmodeSelected", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = t.currentTarget.dataset.value, r = Object(l.i)(null, {mode: s});
                            e.controller.advanceHistory(r), this.setState({playMode: s})
                        }
                    })
                }

                get filterFromUrl() {
                    return this.getAllowedQueryStringValue(p, Object(o.b)().get("filter"), n.a.userPreferences.get("user_list_filter"))
                }

                get playmodeFromUrl() {
                    return this.getAllowedQueryStringValue(m, Object(o.b)().get("mode"), "all")
                }

                get sortedUsers() {
                    const e = this.getFilteredUsers(this.state.filter).slice();
                    switch (this.state.sortMode) {
                        case"rank":
                            return e.sort(b);
                        case"username":
                            return e.sort(f);
                        default:
                            return e.sort((e, t) => e.is_online && t.is_online ? f(e, t) : e.is_online || t.is_online ? e.is_online ? -1 : 1 : r(t.last_visit || 0).diff(r(e.last_visit || 0)))
                    }
                }

                get sortFromUrl() {
                    return this.getAllowedQueryStringValue(h, Object(o.b)().get("sort"), n.a.userPreferences.get("user_list_sort"))
                }

                get viewFromUrl() {
                    return this.getAllowedQueryStringValue(u.b, Object(o.b)().get("view"), n.a.userPreferences.get("user_list_view"))
                }

                render() {
                    return i.createElement(i.Fragment, null, this.renderSelections(), i.createElement("div", {className: "user-list"}, null != this.props.title && i.createElement("h1", {className: "user-list__title"}, this.props.title), null != this.props.descriptionHtml && i.createElement("div", {
                        dangerouslySetInnerHTML: {__html: this.props.descriptionHtml},
                        className: "user-list__description"
                    }), i.createElement("div", {className: "user-list__toolbar"}, this.props.playmodeFilter && i.createElement("div", {className: "user-list__toolbar-row"}, i.createElement("div", {className: "user-list__toolbar-item"}, this.renderPlaymodeFilter())), i.createElement("div", {className: "user-list__toolbar-row"}, i.createElement("div", {className: "user-list__toolbar-item"}, this.renderSorter()), i.createElement("div", {className: "user-list__toolbar-item"}, this.renderViewMode()))), i.createElement("div", {className: "user-list__items"}, i.createElement(d.a, {
                        users: this.sortedUsers,
                        viewMode: this.state.viewMode
                    }))))
                }

                renderOption(e, t, s = !1) {
                    const r = s ? ["active"] : [];
                    let n = Object(a.a)("update-streams-v2__item", r);
                    return n += ` t-changelog-stream--${e}`, i.createElement("a", {
                        key: e,
                        className: n,
                        "data-key": e,
                        href: Object(l.i)(null, {filter: e}),
                        onClick: this.optionSelected
                    }, i.createElement("div", {className: "update-streams-v2__bar u-changelog-stream--bg"}), i.createElement("p", {className: "update-streams-v2__row update-streams-v2__row--name"}, osu.trans(`users.status.${e}`)), i.createElement("p", {className: "update-streams-v2__row update-streams-v2__row--version"}, t))
                }

                renderSelections() {
                    return i.createElement("div", {className: "update-streams-v2 update-streams-v2--with-active update-streams-v2--user-list"}, i.createElement("div", {className: "update-streams-v2__container"}, p.map(e => this.renderOption(e, this.getFilteredUsers(e).length, e === this.state.filter))))
                }

                renderSorter() {
                    return i.createElement(c.a, {
                        currentValue: this.state.sortMode,
                        modifiers: ["user-list"],
                        onChange: this.handleSortChange,
                        values: h
                    })
                }

                renderViewMode() {
                    return i.createElement("div", {className: "user-list__view-modes"}, i.createElement("button", {
                        className: Object(a.a)("user-list__view-mode", "card" === this.state.viewMode ? ["active"] : []),
                        "data-value": "card",
                        onClick: this.onViewSelected,
                        title: osu.trans("users.view_mode.card")
                    }, i.createElement("span", {className: "fas fa-square"})), i.createElement("button", {
                        className: Object(a.a)("user-list__view-mode", "list" === this.state.viewMode ? ["active"] : []),
                        "data-value": "list",
                        onClick: this.onViewSelected,
                        title: osu.trans("users.view_mode.list")
                    }, i.createElement("span", {className: "fas fa-bars"})), i.createElement("button", {
                        className: Object(a.a)("user-list__view-mode", "brick" === this.state.viewMode ? ["active"] : []),
                        "data-value": "brick",
                        onClick: this.onViewSelected,
                        title: osu.trans("users.view_mode.brick")
                    }, i.createElement("span", {className: "fas fa-th"})))
                }

                getAllowedQueryStringValue(e, t, s) {
                    const r = t;
                    if (e.indexOf(r) > -1) return r;
                    const n = s;
                    return e.indexOf(n) > -1 ? n : e[0]
                }

                getFilteredUsers(e) {
                    let t = this.props.users.slice();
                    const s = this.state.playMode;
                    switch (this.props.playmodeFilter && "all" !== s && (t = t.filter(e => {
                        var t;
                        if (e.groups && e.groups.length > 0) {
                            if (null != this.props.playmodeFilterGroupId) {
                                const r = e.groups.find(e => e.id === this.props.playmodeFilterGroupId);
                                return null === (t = null == r ? void 0 : r.playmodes) || void 0 === t ? void 0 : t.includes(s)
                            }
                            return e.groups.some(e => {
                                var t;
                                return null === (t = e.playmodes) || void 0 === t ? void 0 : t.includes(s)
                            })
                        }
                        return !1
                    })), e) {
                        case"online":
                            return t.filter(e => e.is_online);
                        case"offline":
                            return t.filter(e => !e.is_online);
                        default:
                            return t
                    }
                }

                renderPlaymodeFilter() {
                    const e = m.map(e => i.createElement("button", {
                        key: e,
                        className: Object(a.a)("user-list__view-mode", this.state.playMode === e ? ["active"] : []),
                        "data-value": e,
                        onClick: this.playmodeSelected,
                        title: osu.trans(`beatmaps.mode.${e}`)
                    }, "all" === e ? i.createElement("span", null, osu.trans("beatmaps.mode.all")) : i.createElement("span", {className: `fal fa-extra-mode-${e}`})));
                    return i.createElement("div", {className: "user-list__view-modes"}, i.createElement("span", {className: "user-list__view-mode-title"}, osu.trans("users.filtering.by_game_mode")), " ", e)
                }
            }
        }).call(this, s("dMdw"))
    }, "8gxX": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("Hs9Z");

        class n {
            constructor(e = 7500, t = 18e5) {
                Object.defineProperty(this, "initialDelay", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "maxDelay", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: t
                }), Object.defineProperty(this, "current", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), this.current = Math.max(e, 1)
            }

            get() {
                const e = this.current + Object(r.random)(5e3, 2e4);
                return this.current = Math.min(2 * this.current, this.maxDelay), e
            }

            reset() {
                this.current = this.initialDelay
            }
        }
    }, "8sf4": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return O
            }));
            var r = s("OjW+"), n = s("IwO5"), i = s("pdUJ"), a = s("0h6b"), o = s("Hs9Z"), l = s("lv9K"), c = s("KUml"),
                u = s("f4vq"), d = s("/G9H"), p = s("X4fg"), m = s("yun6"), h = s("ubBH"), b = s("tX/w"), f = s("phBA"),
                v = s("vZz4"), g = s("55pz"), y = s("UBw1"), w = s("yHuj"), _ = function (e, t, s, r) {
                    var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };
            const O = ["normal", "extra"], j = {
                    approved: "ranked_date",
                    graveyard: "last_updated",
                    loved: "ranked_date",
                    pending: "last_updated",
                    qualified: "ranked_date",
                    ranked: "ranked_date",
                    wip: "last_updated"
                }, E = Object(c.b)(({beatmap: e}) => d.createElement("div", {
                    className: "beatmapset-panel__beatmap-dot",
                    style: {"--bg": Object(m.d)(e.difficulty_rating)}
                })), P = Object(c.b)(({
                                          compact: e,
                                          beatmaps: t,
                                          mode: s
                                      }) => d.createElement("div", {className: "beatmapset-panel__extra-item beatmapset-panel__extra-item--dots"}, d.createElement("div", {className: "beatmapset-panel__beatmap-icon"}, d.createElement("i", {className: `fal fa-extra-mode-${s}`})), e ? d.createElement("div", {className: "beatmapset-panel__beatmap-count"}, t.length) : t.map(e => d.createElement(E, {
                    key: e.id,
                    beatmap: e
                })))),
                k = () => d.createElement("span", {className: "beatmapset-badge beatmapset-badge--featured-artist beatmapset-badge--panel"}, osu.trans("beatmapsets.featured_artist_badge.label")),
                N = Object(c.b)(({beatmapset: e}) => d.createElement(w.a, {
                    className: "beatmapset-panel__mapper-link u-hover",
                    user: {id: e.user_id, username: e.creator}
                })),
                S = () => d.createElement("span", {className: "beatmapset-badge beatmapset-badge--nsfw beatmapset-badge--panel"}, osu.trans("beatmapsets.nsfw_badge.label")),
                T = ({icon: e, titleVariant: t}) => d.createElement("div", {
                    className: "beatmapset-panel__play-icon",
                    title: osu.trans(`beatmapsets.show.info.${t}`)
                }, d.createElement("i", {className: e})), C = ({
                                                                   icon: e,
                                                                   title: t,
                                                                   type: s,
                                                                   value: r
                                                               }) => d.createElement("div", {
                    className: `beatmapset-panel__stats-item beatmapset-panel__stats-item--${s}`,
                    title: t
                }, d.createElement("span", {className: "beatmapset-panel__stats-item-icon"}, d.createElement("i", {className: `fa-fw ${e}`})), d.createElement("span", null, Object(f.d)(r, void 0, {
                    maximumFractionDigits: 1,
                    minimumFractionDigits: 0
                })));
            let x = class extends d.Component {
                constructor(t) {
                    super(t), Object.defineProperty(this, "beatmapsPopupHover", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "beatmapsPopupRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: d.createRef()
                    }), Object.defineProperty(this, "blockRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: d.createRef()
                    }), Object.defineProperty(this, "mobileExpanded", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "timeouts", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "beatmapsPopupDelayedHide", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timeouts.beatmapsPopup), this.beatmapsPopupHover && (this.timeouts.beatmapsPopup = window.setTimeout(Object(l.f)(() => {
                                this.beatmapsPopupHover = !1
                            }), 500))
                        }
                    }), Object.defineProperty(this, "beatmapsPopupDelayedShow", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timeouts.beatmapsPopup), this.beatmapsPopupHover || (this.timeouts.beatmapsPopup = window.setTimeout(Object(l.f)(() => {
                                this.beatmapsPopupHover = !0
                            }), 100))
                        }
                    }), Object.defineProperty(this, "beatmapsPopupHide", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timeouts.beatmapsPopup), this.beatmapsPopupHover = !1
                        }
                    }), Object.defineProperty(this, "beatmapsPopupKeep", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timeouts.beatmapsPopup), this.beatmapsPopupHover = !0
                        }
                    }), Object.defineProperty(this, "onBeatmapsPopupEnter", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.beatmapsPopupKeep()
                        }
                    }), Object.defineProperty(this, "onBeatmapsPopupLeave", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.beatmapsPopupDelayedHide()
                        }
                    }), Object.defineProperty(this, "onDocumentClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            var s, r, n;
                            this.mobileExpanded && (null !== (s = this.blockRef.current) && void 0 !== s && s.contains(t.target) || null !== (n = null === (r = this.beatmapsPopupRef.current) || void 0 === r ? void 0 : r.contentRef.current) && void 0 !== n && n.contains(t.target) || (e(document).off("click", this.onDocumentClick), this.mobileExpanded = !1))
                        }
                    }), Object.defineProperty(this, "onExtraRowEnter", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.beatmapsPopupDelayedShow()
                        }
                    }), Object.defineProperty(this, "onExtraRowLeave", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.beatmapsPopupDelayedHide()
                        }
                    }), Object.defineProperty(this, "onMobileExpandToggleClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.mobileExpanded = !this.mobileExpanded, this.mobileExpanded && e(document).on("click", this.onDocumentClick)
                        }
                    }), Object.defineProperty(this, "toggleFavourite", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            Object(h.c)(this.props.beatmapset)
                        }
                    }), Object(l.p)(this)
                }

                get beatmapDotsCompact() {
                    return null != this.props.beatmapset.beatmaps && this.props.beatmapset.beatmaps.length > 12
                }

                get displayDate() {
                    const e = j[this.props.beatmapset.status];
                    return this.props.beatmapset[e]
                }

                get downloadLink() {
                    var e;
                    if (null == u.a.currentUser) return {title: osu.trans("beatmapsets.show.details.logged-out")};
                    if (null === (e = this.props.beatmapset.availability) || void 0 === e ? void 0 : e.download_disabled) return {title: osu.trans("beatmapsets.availability.disabled")};
                    let t, s, r = u.a.userPreferences.get("beatmapset_download");
                    if ("direct" !== r || u.a.currentUser.is_supporter || (r = "all"), "direct" === r) t = Object(v.b)(this.props.beatmapset.id), s = "direct"; else {
                        const e = {beatmapset: this.props.beatmapset.id};
                        this.props.beatmapset.video ? "no_video" === r ? (e.noVideo = 1, s = "no_video") : s = "video" : s = "all", t = Object(a.a)("beatmapsets.download", e)
                    }
                    return {title: osu.trans(`beatmapsets.panel.download.${s}`), url: t}
                }

                get favourite() {
                    return this.props.beatmapset.has_favourited ? {
                        icon: "fas fa-heart",
                        toggleTitleVariant: "unfavourite"
                    } : {icon: "far fa-heart", toggleTitleVariant: "favourite"}
                }

                get groupedBeatmaps() {
                    return Object(m.f)(this.props.beatmapset.beatmaps)
                }

                get isBeatmapsPopupVisible() {
                    return this.beatmapsPopupHover || this.mobileExpanded
                }

                get nominations() {
                    return null != this.props.beatmapset.nominations_summary ? this.props.beatmapset.nominations_summary : null != this.props.beatmapset.nominations ? this.props.beatmapset.nominations.legacy_mode ? this.props.beatmapset.nominations : {
                        current: Object(o.sum)(Object(o.values)(this.props.beatmapset.nominations.current)),
                        required: Object(o.sum)(Object(o.values)(this.props.beatmapset.nominations.required))
                    } : void 0
                }

                get showHypeCounts() {
                    return null != this.props.beatmapset.hype
                }

                get showVisual() {
                    return Object(h.b)(this.props.beatmapset)
                }

                get url() {
                    return Object(a.a)("beatmapsets.show", {beatmapset: this.props.beatmapset.id})
                }

                componentWillUnmount() {
                    e(document).off("click", this.onDocumentClick), Object.values(this.timeouts).forEach(e => {
                        window.clearTimeout(e)
                    })
                }

                render() {
                    let e = Object(b.a)("beatmapset-panel", {
                        "beatmaps-popup-visible": this.isBeatmapsPopupVisible,
                        "mobile-expanded": this.mobileExpanded,
                        [`size-${u.a.userPreferences.get("beatmapset_card_size")}`]: !0,
                        "with-hype-counts": this.showHypeCounts
                    });
                    return this.showVisual && (e += " js-audio--player"), d.createElement("div", {
                        ref: this.blockRef,
                        className: e,
                        "data-audio-url": this.props.beatmapset.preview_url,
                        onMouseLeave: this.beatmapsPopupHide,
                        style: {"--beatmaps-popup-transition-duration": "150ms"}
                    }, this.renderBeatmapsPopup(), this.renderCover(), d.createElement("div", {className: "beatmapset-panel__content"}, this.renderPlayArea(), this.renderInfoArea(), this.renderMenuArea()), d.createElement("button", {
                        className: "beatmapset-panel__mobile-expand",
                        onClick: this.onMobileExpandToggleClick,
                        type: "button"
                    }, d.createElement("span", {className: `fas fa-angle-${this.mobileExpanded ? "up" : "down"}`})))
                }

                renderBeatmapsPopup() {
                    return d.createElement(p.a, {
                        in: this.isBeatmapsPopupVisible,
                        mountOnEnter: !0,
                        timeout: {enter: 0, exit: 150},
                        unmountOnExit: !0
                    }, e => d.createElement(r.a, {
                        ref: this.beatmapsPopupRef,
                        groupedBeatmaps: this.groupedBeatmaps,
                        onMouseEnter: this.onBeatmapsPopupEnter,
                        onMouseLeave: this.onBeatmapsPopupLeave,
                        parent: this.blockRef.current,
                        state: e,
                        transitionDuration: 150
                    }))
                }

                renderCover() {
                    return d.createElement("a", {
                        className: "beatmapset-panel__cover-container",
                        href: this.url
                    }, d.createElement("div", {className: "beatmapset-panel__cover-col beatmapset-panel__cover-col--play"}, d.createElement("div", {className: "beatmapset-panel__cover beatmapset-panel__cover--default"}), this.showVisual && d.createElement(i.a, {
                        className: "beatmapset-panel__cover",
                        hideOnError: !0,
                        src: this.props.beatmapset.covers.list
                    })), d.createElement("div", {className: "beatmapset-panel__cover-col beatmapset-panel__cover-col--info"}, d.createElement("div", {className: "beatmapset-panel__cover beatmapset-panel__cover--default"}), this.showVisual && u.a.windowSize.isDesktop && d.createElement(i.a, {
                        className: "beatmapset-panel__cover",
                        hideOnError: !0,
                        src: this.props.beatmapset.covers.card
                    })))
                }

                renderInfoArea() {
                    return d.createElement("div", {className: "beatmapset-panel__info"}, d.createElement("div", {className: "beatmapset-panel__info-row beatmapset-panel__info-row--title"}, d.createElement("a", {
                        className: "beatmapset-panel__main-link u-ellipsis-overflow",
                        href: this.url
                    }, Object(m.e)(this.props.beatmapset)), this.props.beatmapset.nsfw && d.createElement(S, null)), d.createElement("div", {className: "beatmapset-panel__info-row beatmapset-panel__info-row--artist"}, d.createElement("a", {
                        className: "beatmapset-panel__main-link u-ellipsis-overflow",
                        href: this.url
                    }, osu.trans("beatmapsets.show.details.by_artist", {artist: Object(m.c)(this.props.beatmapset)})), null != this.props.beatmapset.track_id && d.createElement(k, null)), d.createElement("div", {className: "beatmapset-panel__info-row beatmapset-panel__info-row--source"}, d.createElement("div", {className: "u-ellipsis-overflow"}, this.props.beatmapset.source)), d.createElement("div", {className: "beatmapset-panel__info-row beatmapset-panel__info-row--mapper"}, d.createElement("div", {className: "u-ellipsis-overflow"}, d.createElement(g.a, {
                        mappings: {mapper: d.createElement(N, {beatmapset: this.props.beatmapset})},
                        pattern: osu.trans("beatmapsets.show.details.mapped_by")
                    }))), d.createElement("div", {className: "beatmapset-panel__info-row beatmapset-panel__info-row--stats"}, this.showHypeCounts && null != this.props.beatmapset.hype && d.createElement(C, {
                        icon: "fas fa-bullhorn",
                        title: osu.trans("beatmaps.hype.required_text", {
                            current: osu.formatNumber(this.props.beatmapset.hype.current),
                            required: osu.formatNumber(this.props.beatmapset.hype.required)
                        }),
                        type: "hype",
                        value: this.props.beatmapset.hype.current
                    }), this.showHypeCounts && null != this.nominations && d.createElement(C, {
                        icon: "fas fa-thumbs-up",
                        title: osu.trans("beatmaps.nominations.required_text", {
                            current: osu.formatNumber(this.nominations.current),
                            required: osu.formatNumber(this.nominations.required)
                        }),
                        type: "nominations",
                        value: this.nominations.current
                    }), d.createElement(C, {
                        icon: this.favourite.icon,
                        title: osu.trans("beatmaps.panel.favourites", {count: osu.formatNumber(this.props.beatmapset.favourite_count)}),
                        type: "favourite-count",
                        value: this.props.beatmapset.favourite_count
                    }), d.createElement(C, {
                        icon: "fas fa-play-circle",
                        title: osu.trans("beatmaps.panel.playcount", {count: osu.formatNumber(this.props.beatmapset.play_count)}),
                        type: "play-count",
                        value: this.props.beatmapset.play_count
                    }), d.createElement("div", {className: "beatmapset-panel__stats-item beatmapset-panel__stats-item--date"}, d.createElement("span", {className: "beatmapset-panel__stats-item-icon"}, d.createElement("i", {className: "fa-fw fas fa-check-circle"})), d.createElement(y.a, {dateTime: this.displayDate}))), d.createElement("a", {
                        className: "beatmapset-panel__info-row beatmapset-panel__info-row--extra",
                        href: this.url,
                        onMouseEnter: this.onExtraRowEnter,
                        onMouseLeave: this.onExtraRowLeave
                    }, d.createElement("div", {className: "beatmapset-panel__extra-item"}, d.createElement("div", {
                        className: "beatmapset-status beatmapset-status--panel",
                        style: {
                            "--bg": `var(--beatmapset-${this.props.beatmapset.status}-bg)`,
                            "--colour": `var(--beatmapset-${this.props.beatmapset.status}-colour)`
                        }
                    }, osu.trans(`beatmapsets.show.status.${this.props.beatmapset.status}`))), [...this.groupedBeatmaps].map(([e, t]) => t.length > 0 && d.createElement(P, {
                        key: e,
                        beatmaps: t,
                        compact: this.beatmapDotsCompact,
                        mode: e
                    }))))
                }

                renderMenuArea() {
                    return d.createElement("div", {className: "beatmapset-panel__menu-container"}, d.createElement("div", {className: "beatmapset-panel__menu"}, null == u.a.currentUser ? d.createElement("span", {
                        className: "beatmapset-panel__menu-item beatmapset-panel__menu-item--disabled",
                        title: osu.trans("beatmapsets.show.details.favourite_login")
                    }, d.createElement("span", {className: this.favourite.icon})) : d.createElement("button", {
                        className: "beatmapset-panel__menu-item",
                        onClick: this.toggleFavourite,
                        title: osu.trans(`beatmapsets.show.details.${this.favourite.toggleTitleVariant}`),
                        type: "button"
                    }, d.createElement("span", {className: this.favourite.icon})), null == this.downloadLink.url ? d.createElement("span", {
                        className: "beatmapset-panel__menu-item beatmapset-panel__menu-item--disabled",
                        title: this.downloadLink.title
                    }, d.createElement("span", {className: "fas fa-file-download"})) : d.createElement("a", {
                        className: "beatmapset-panel__menu-item",
                        "data-turbolinks": "false",
                        href: this.downloadLink.url,
                        title: this.downloadLink.title
                    }, d.createElement("span", {className: "fas fa-file-download"}))))
                }

                renderPlayArea() {
                    return d.createElement("div", {className: "beatmapset-panel__play-container"}, this.showVisual && d.createElement("button", {
                        className: "beatmapset-panel__play js-audio--play",
                        type: "button"
                    }, d.createElement("span", {className: "play-button"})), d.createElement("div", {className: "beatmapset-panel__play-progress"}, d.createElement(n.a, {
                        current: 0,
                        ignoreProgress: !0,
                        max: 1,
                        onlyShowAsWarning: !1,
                        theme: "beatmapset-panel"
                    })), d.createElement("div", {className: "beatmapset-panel__play-icons"}, this.props.beatmapset.video && d.createElement(T, {
                        icon: "fas fa-film",
                        titleVariant: "video"
                    }), this.props.beatmapset.storyboard && d.createElement(T, {
                        icon: "fas fa-image",
                        titleVariant: "storyboard"
                    })))
                }
            };
            _([l.q], x.prototype, "beatmapsPopupHover", void 0), _([l.q], x.prototype, "mobileExpanded", void 0), _([l.h], x.prototype, "beatmapDotsCompact", null), _([l.h], x.prototype, "displayDate", null), _([l.h], x.prototype, "downloadLink", null), _([l.h], x.prototype, "favourite", null), _([l.h], x.prototype, "groupedBeatmaps", null), _([l.h], x.prototype, "isBeatmapsPopupVisible", null), _([l.h], x.prototype, "nominations", null), _([l.h], x.prototype, "showHypeCounts", null), _([l.h], x.prototype, "showVisual", null), _([l.h], x.prototype, "url", null), _([l.f], x.prototype, "beatmapsPopupHide", void 0), _([l.f], x.prototype, "beatmapsPopupKeep", void 0), _([l.f], x.prototype, "onDocumentClick", void 0), _([l.f], x.prototype, "onMobileExpandToggleClick", void 0), x = _([c.b], x), t.b = x
        }).call(this, s("5wds"))
    }, "9CpU": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return l
        }));
        var r, n = s("0h6b"), i = s("/G9H"), a = s("I8Ok"), o = s("vZz4");
        i.createElement, r = "beatmap-discussion-system-post";
        var l = function (e) {
            var t, s;
            return t = function () {
                switch (e.post.message.type) {
                    case"resolved":
                        return osu.trans("beatmap_discussions.system.resolved." + e.post.message.value, {user: Object(o.f)(Object(n.a)("users.show", {user: e.user.id}), e.user.username, {classNames: [r + "__user"]})})
                }
            }(), s = r + " " + r + "--" + e.post.message.type, e.post.deleted_at && (s += " " + r + "--deleted"), Object(a.div)({className: s}, Object(a.div)({
                className: r + "__content",
                dangerouslySetInnerHTML: {__html: t}
            }))
        }
    }, "9zVE": function (e, t, s) {
        "use strict";
        var r = s("/HbY"), n = s("KUml"), i = s("/G9H"), a = s("tX/w"), o = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let l = class extends i.Component {
            render() {
                return this.props.isMarkingAsRead ? i.createElement("div", {className: Object(a.a)("notification-action-button", this.props.modifiers)}, i.createElement("span", {className: "notification-action-button__text"}, this.props.text), i.createElement("div", {className: "notification-action-button__icon"}, i.createElement(r.a, null))) : i.createElement("button", {
                    className: Object(a.a)("notification-action-button", this.props.modifiers),
                    onClick: this.props.onMarkAsRead,
                    type: "button"
                }, i.createElement("span", {className: "notification-action-button__text"}, this.props.text), i.createElement("div", {className: "notification-action-button__icon"}, i.createElement("span", {className: "fas fa-check"})))
            }
        };
        Object.defineProperty(l, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {modifiers: []}
        }), l = o([n.b], l), t.a = l
    }, AqrC: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("LPNJ"), n = s("/G9H"), i = s("WKXC"), a = s("74hk"), o = s("cX0L");

            class l extends n.PureComponent {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "state", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {active: !1}
                    }), Object.defineProperty(this, "buttonRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.createRef()
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `popup-menu-${Object(o.a)()}`
                    }), Object.defineProperty(this, "menuRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.createRef()
                    }), Object.defineProperty(this, "portal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: document.createElement("div")
                    }), Object.defineProperty(this, "tooltipHideEvent", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "addPortal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null == this.portal.parentElement && document.body.appendChild(this.portal)
                        }
                    }), Object.defineProperty(this, "dismiss", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setState({active: !1})
                        }
                    }), Object.defineProperty(this, "hide", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (!this.state.active || Object(r.a)()) return;
                            const t = e.originalEvent;
                            null != t && ("key" in t && "Escape" === t.key || "button" in t && 0 === t.button && !this.isMenuInPath(t.composedPath())) && this.setState({active: !1})
                        }
                    }), Object.defineProperty(this, "removePortal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.portal.remove()
                        }
                    }), Object.defineProperty(this, "resize", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (!this.state.active) return;
                            if (null == this.buttonRef.current || null == this.menuRef.current) throw new Error("missing button and/or menu element");
                            if (null == this.buttonRef.current.offsetParent) return void (this.portal.style.display = "none");
                            const e = this.buttonRef.current.getBoundingClientRect(),
                                t = this.menuRef.current.getBoundingClientRect(), {scrollX: s, scrollY: r} = window;
                            let n = s + e.right;
                            ("right" === this.props.direction || e.x - t.width < 0) && (n += t.width - e.width), this.portal.style.display = "block", this.portal.style.position = "absolute", this.portal.style.top = `${Math.floor(r + e.bottom + 5)}px`, this.portal.style.left = `${Math.floor(n)}px`;
                            const i = this.tooltipElement;
                            null != i && (this.portal.style.zIndex = getComputedStyle(i).zIndex)
                        }
                    }), Object.defineProperty(this, "toggle", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setState({active: !this.state.active})
                        }
                    })
                }

                get $tooltipElement() {
                    const t = this.tooltipElement;
                    return null == t ? null : e(t)
                }

                get tooltipElement() {
                    var e;
                    return null === (e = this.context) || void 0 === e ? void 0 : e.closest(".qtip")
                }

                componentDidMount() {
                    var t;
                    this.tooltipHideEvent = null === (t = this.$tooltipElement) || void 0 === t ? void 0 : t.qtip("option", "hide.event"), e(window).on(`resize.${this.eventId}`, this.resize), e(document).on(`turbolinks:before-cache.${this.eventId}`, this.removePortal)
                }

                componentDidUpdate(t, s) {
                    var r, n, i, a, o;
                    if (s.active !== this.state.active) if (this.state.active) {
                        this.addPortal(), this.resize();
                        const t = this.$tooltipElement;
                        null != t && (this.tooltipHideEvent = t.qtip("option", "hide.event"), t.qtip("option", "hide.event", !1)), e(document).on(`click.${this.eventId} keydown.${this.eventId}`, this.hide), null === (n = (r = this.props).onShow) || void 0 === n || n.call(r)
                    } else this.removePortal(), null === (i = this.$tooltipElement) || void 0 === i || i.qtip("option", "hide.event", this.tooltipHideEvent), e(document).off(`click.${this.eventId} keydown.${this.eventId}`, this.hide), null === (o = (a = this.props).onHide) || void 0 === o || o.call(a)
                }

                componentWillUnmount() {
                    e(document).off(`.${this.eventId}`), e(window).off(`.${this.eventId}`)
                }

                render() {
                    return this.props.customRender ? this.props.customRender(Object(i.createPortal)(this.renderMenu(), this.portal), this.buttonRef, this.toggle) : n.createElement(n.Fragment, null, n.createElement("button", {
                        ref: this.buttonRef,
                        className: "popup-menu",
                        onClick: this.toggle,
                        type: "button"
                    }, n.createElement("span", {className: "fas fa-ellipsis-v"})), Object(i.createPortal)(this.renderMenu(), this.portal))
                }

                isMenuInPath(e) {
                    return e.includes(this.portal) || null != this.buttonRef.current && e.includes(this.buttonRef.current)
                }

                renderMenu() {
                    return this.state.active ? n.createElement("div", {
                        ref: this.menuRef,
                        className: "popup-menu-float"
                    }, this.props.children(this.dismiss)) : null
                }
            }

            Object.defineProperty(l, "contextType", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: a.a
            }), Object.defineProperty(l, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {children: () => null, direction: "left"}
            })
        }).call(this, s("5wds"))
    }, B5xN: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("2qTP"), n = s("0h6b"), i = s("/G9H"), a = function (e, t) {
                var s = {};
                for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (s[r] = e[r]);
                if (null != e && "function" == typeof Object.getOwnPropertySymbols) {
                    var n = 0;
                    for (r = Object.getOwnPropertySymbols(e); n < r.length; n++) t.indexOf(r[n]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[n]) && (s[r[n]] = e[r[n]])
                }
                return s
            };
            const o = {
                beatmapset: ["UnwantedContent", "Other"],
                beatmapset_discussion_post: ["Insults", "Spam", "UnwantedContent", "Nonsense", "Other"],
                comment: ["Insults", "Spam", "UnwantedContent", "Nonsense", "Other"],
                forum_post: ["Insults", "Spam", "UnwantedContent", "Nonsense", "Other"],
                scores: ["Cheating", "MultipleAccounts", "Other"]
            };

            class l extends i.PureComponent {
                constructor(t) {
                    super(t), Object.defineProperty(this, "timeout", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "onFormClose", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setState({disabled: !1, showingForm: !1}, this.props.onFormClose)
                        }
                    }), Object.defineProperty(this, "onSubmit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            this.setState({disabled: !0});
                            const s = {
                                data: {
                                    comments: t.comments,
                                    reason: t.reason,
                                    reportable_id: this.props.reportableId,
                                    reportable_type: this.props.reportableType
                                }, dataType: "json", type: "POST", url: Object(n.a)("reports.store")
                            };
                            e.ajax(s).done(() => {
                                this.timeout = window.setTimeout(this.onFormClose, 1e3), this.setState({completed: !0})
                            }).fail(e => {
                                osu.ajaxError(e), this.setState({disabled: !1})
                            })
                        }
                    }), Object.defineProperty(this, "showForm", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timeout), this.setState({disabled: !1, showingForm: !0})
                        }
                    }), Object.defineProperty(this, "onShowFormButtonClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            0 === e.button && (e.preventDefault(), this.showForm())
                        }
                    }), this.state = {completed: !1, disabled: !1, showingForm: !1}
                }

                render() {
                    const e = this.props, {
                            baseKey: t,
                            icon: s,
                            onFormClose: n,
                            reportableId: l,
                            reportableType: c,
                            user: u
                        } = e, d = a(e, ["baseKey", "icon", "onFormClose", "reportableId", "reportableType", "user"]),
                        p = t || this.props.reportableType;
                    return i.createElement(i.Fragment, null, i.createElement("button", Object.assign({
                        key: "button",
                        onClick: this.onShowFormButtonClick
                    }, d), s ? i.createElement("span", {className: "textual-button textual-button--inline"}, i.createElement("i", {className: "textual-button__icon fas fa-exclamation-triangle"}), " ", osu.trans(`report.${p}.button`)) : osu.trans(`report.${p}.button`)), this.state.showingForm && i.createElement(r.a, {
                        completed: this.state.completed,
                        disabled: this.state.disabled,
                        onClose: this.onFormClose,
                        onSubmit: this.onSubmit,
                        title: osu.trans(`report.${p}.title`, {username: `<strong>${u.username}</strong>`}),
                        visible: !0,
                        visibleOptions: o[p]
                    }))
                }
            }

            Object.defineProperty(l, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {
                    icon: !1, onFormClose: () => {
                    }
                }
            })
        }).call(this, s("5wds"))
    }, BCxl: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return m
            }));
            var r = s("Hs9Z"), n = s("/G9H"), i = s("tSlR"), a = s("dTpI"), o = s("tX/w"), l = s("vZz4"), c = s("5AHq"),
                u = s("yJmy"), d = s("zr5c"), p = s("ss8h");

            class m extends n.Component {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "bn", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "beatmap-discussion-review-post-embed-preview"
                    }), Object.defineProperty(this, "cache", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "tooltipContent", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.createRef()
                    }), Object.defineProperty(this, "tooltipEl", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "componentDidMount", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.editable() && i.h.setNodes(this.context, {timestamp: void 0}, {at: this.path()})
                        }
                    }), Object.defineProperty(this, "componentDidUpdate", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (!this.editable()) return;
                            const e = this.path();
                            let t = !1;
                            if (this.props.element.beatmapId) {
                                const s = this.props.element.children[0].text,
                                    r = BeatmapDiscussionHelper.TIMESTAMP_REGEX.exec(s);
                                let n;
                                null !== r && 0 === r.index && (n = r[2]), n !== this.props.element.timestamp && (t = !0), i.h.setNodes(this.context, {timestamp: n}, {at: e})
                            } else i.h.setNodes(this.context, {timestamp: void 0}, {at: e}), t = !0;
                            t && (this.cache = {}, this.destroyTooltip())
                        }
                    }), Object.defineProperty(this, "createTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = this.timestamp();
                            if (null == s) return;
                            const r = t.currentTarget, n = `${this.selectedBeatmap()}-${s}`;
                            this.tooltipEl && this.tooltipEl._tooltip === n || (this.tooltipEl = r, r._tooltip = n, e(r).qtip({
                                content: {
                                    text: () => {
                                        var e;
                                        return null === (e = this.tooltipContent.current) || void 0 === e ? void 0 : e.innerHTML
                                    }
                                },
                                hide: {delay: 200, fixed: !0},
                                position: {at: "top center", my: "bottom center", viewport: e(window)},
                                show: {delay: 200, ready: !0},
                                style: {classes: "tooltip-default tooltip-default--interactable"}
                            }), this.tooltipEl = r)
                        }
                    }), Object.defineProperty(this, "delete", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.setTimeout(() => i.h.delete(this.context, {at: this.path()}), 0)
                        }
                    }), Object.defineProperty(this, "destroyTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (!this.tooltipEl) return;
                            const t = e(this.tooltipEl).qtip("api");
                            t && (t.destroy(), this.tooltipEl = void 0)
                        }
                    }), Object.defineProperty(this, "discussionType", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => this.props.element.discussionType
                    }), Object.defineProperty(this, "editable", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => !(this.props.editMode && this.props.element.discussionId)
                    }), Object.defineProperty(this, "isRelevantDiscussion", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => null != e && e.beatmap_id === this.selectedBeatmap()
                    }), Object.defineProperty(this, "nearbyDiscussions", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e;
                            const t = this.timestamp();
                            if (null == t) return [];
                            if (!this.cache.nearbyDiscussions || this.cache.nearbyDiscussions.timestamp !== t || this.cache.nearbyDiscussions.beatmap_id !== this.selectedBeatmap()) {
                                const e = r.filter(this.props.discussions, this.isRelevantDiscussion);
                                this.cache.nearbyDiscussions = {
                                    beatmap_id: this.selectedBeatmap(),
                                    discussions: BeatmapDiscussionHelper.nearbyDiscussions(e, t),
                                    timestamp: t
                                }
                            }
                            return null === (e = this.cache.nearbyDiscussions) || void 0 === e ? void 0 : e.discussions
                        }
                    }), Object.defineProperty(this, "nearbyDraftEmbeds", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = this.timestamp();
                            if (null != t && 0 !== e.length) return e.filter(e => {
                                if (!e.timestamp || e.beatmapId !== this.props.element.beatmapId) return !1;
                                const s = BeatmapDiscussionHelper.parseTimestamp(e.timestamp);
                                return null != s && Math.abs(s - t) <= 5e3
                            })
                        }
                    }), Object.defineProperty(this, "nearbyIndicator", {
                        enumerable: !0, configurable: !0, writable: !0, value: e => {
                            if (null == this.timestamp() || "praise" === this.discussionType()) return;
                            const t = this.editable() ? this.nearbyDiscussions() : [],
                                s = this.nearbyDraftEmbeds(e) || [];
                            if (t.length > 0 || s.length > 1) {
                                const e = t.map(e => {
                                    const t = BeatmapDiscussionHelper.formatTimestamp(e.timestamp);
                                    if (null != t) return this.props.editMode ? t : Object(l.f)(BeatmapDiscussionHelper.url({discussion: e}), t, {classNames: ["js-beatmap-discussion--jump"]})
                                });
                                s.length > 1 && e.push(osu.trans("beatmap_discussions.nearby_posts.unsaved", {count: s.length - 1}));
                                const r = osu.transArray(e), i = osu.trans("beatmap_discussions.nearby_posts.notice", {
                                    existing_timestamps: r,
                                    timestamp: this.props.element.timestamp
                                });
                                return n.createElement("div", {
                                    className: `${this.bn}__indicator ${this.bn}__indicator--warning`,
                                    contentEditable: !1,
                                    onMouseOver: this.createTooltip,
                                    onTouchStart: this.createTooltip
                                }, n.createElement("script", {
                                    dangerouslySetInnerHTML: {__html: i},
                                    ref: this.tooltipContent,
                                    type: "text/html"
                                }), n.createElement("i", {className: "fas fa-exclamation-triangle"}))
                            }
                        }
                    }), Object.defineProperty(this, "path", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => a.b.findPath(this.context, this.props.element)
                    }), Object.defineProperty(this, "selectedBeatmap", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => this.props.element.beatmapId
                    }), Object.defineProperty(this, "timestamp", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => BeatmapDiscussionHelper.parseTimestamp(this.props.element.timestamp)
                    })
                }

                componentWillUnmount() {
                    this.destroyTooltip()
                }

                render() {
                    const e = this.editable(), t = e ? [] : ["read-only"];
                    let s, r = this.props.element.timestamp;
                    null != this.props.element.beatmapId ? s = "diff" : (s = "all-diff", r = void 0);
                    const i = osu.trans(`beatmaps.discussions.review.embed.timestamp.${s}`, {type: osu.trans(`beatmaps.discussions.message_type.${this.discussionType()}`)}),
                        a = n.createElement("button", {
                            className: `${this.bn}__delete`,
                            contentEditable: !1,
                            disabled: this.props.readOnly,
                            onClick: this.delete,
                            title: osu.trans(`beatmaps.discussions.review.embed.${e ? "delete" : "unlink"}`)
                        }, n.createElement("i", {className: `fas fa-${e ? "trash-alt" : "link"}`})),
                        l = n.createElement(c.a.Consumer, null, this.nearbyIndicator),
                        p = this.props.editMode && e ? n.createElement("div", {
                            className: `${this.bn}__indicator`,
                            contentEditable: !1,
                            title: osu.trans("beatmaps.discussions.review.embed.unsaved")
                        }, n.createElement("i", {className: "fas fa-pencil-alt"})) : null;
                    return n.createElement("div", Object.assign({
                        className: "beatmap-discussion beatmap-discussion--preview",
                        contentEditable: e,
                        suppressContentEditableWarning: !0
                    }, this.props.attributes), n.createElement("div", {className: Object(o.a)(this.bn, t)}, n.createElement("div", {className: `${this.bn}__content`}, n.createElement("div", {
                        className: `${this.bn}__selectors`,
                        contentEditable: !1
                    }, n.createElement(u.a, Object.assign({}, this.props, {disabled: this.props.readOnly || !e})), n.createElement(d.a, Object.assign({}, this.props, {disabled: this.props.readOnly || !e})), n.createElement("div", {
                        className: `${this.bn}__timestamp`,
                        contentEditable: !1
                    }, n.createElement("span", {title: e ? i : ""}, null != r ? r : osu.trans("beatmap_discussions.timestamp_display.general"))), p, l), n.createElement("div", {
                        className: `${this.bn}__stripe`,
                        contentEditable: !1
                    }), n.createElement("div", {className: `${this.bn}__message-container`}, n.createElement("div", {className: "beatmapset-discussion-message"}, this.props.children)), p, l)), a)
                }
            }

            Object.defineProperty(m, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: p.a})
        }).call(this, s("5wds"))
    }, BSzr: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return R
            }));
            var r = s("zFzD"), n = s("nN9y"), i = s("aMFG"), a = s("TyNR"), o = s("mArn"), l = s("eIGF"), c = s("o70V"),
                u = s("+kdZ"), d = s("3J26"), p = s("2Kkx"), m = s("msVt"), h = s("70Tv"), b = s("pTWL"), f = s("kD1C"),
                v = s("iCid"), g = s("3Wjd"), y = s("6se0"), w = s("bPNN"), _ = s("YvoO"), O = s("P9aV"), j = s("zrLC"),
                E = s("4PJi"), P = s("dC65"), k = s("bSvc"), N = s("x6t3"), S = s("lv9K"), T = s("I7Md"), C = s("0VTr"),
                x = s("VsY1"), M = function (e, t, s, r) {
                    var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };

            class R {
                constructor() {
                    Object.defineProperty(this, "beatmapsetSearchController", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "captcha", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new i.a
                    }), Object.defineProperty(this, "chatWorker", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new n.a
                    }), Object.defineProperty(this, "clickMenu", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new a.a
                    }), Object.defineProperty(this, "currentUser", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "currentUserModel", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new O.a(this)
                    }), Object.defineProperty(this, "dataStore", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "enchant", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "forumPoll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new l.a
                    }), Object.defineProperty(this, "forumPostEdit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new c.a
                    }), Object.defineProperty(this, "forumPostInput", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new u.a
                    }), Object.defineProperty(this, "localtime", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new d.a
                    }), Object.defineProperty(this, "mobileToggle", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new p.a
                    }), Object.defineProperty(this, "notificationsWorker", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "osuAudio", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "osuLayzr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new h.a
                    }), Object.defineProperty(this, "reactTurbolinks", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "referenceLinkTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new P.a
                    }), Object.defineProperty(this, "scorePins", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new y.a
                    }), Object.defineProperty(this, "socketWorker", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "stickyHeader", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new f.a
                    }), Object.defineProperty(this, "timeago", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new v.a
                    }), Object.defineProperty(this, "turbolinksReload", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new g.a
                    }), Object.defineProperty(this, "userLogin", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "userLoginObserver", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "userPreferences", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new j.a
                    }), Object.defineProperty(this, "userVerification", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new E.a
                    }), Object.defineProperty(this, "windowFocusObserver", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "windowSize", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new N.a
                    }), Object.defineProperty(this, "onCurrentUserUpdate", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            this.setCurrentUser(t)
                        }
                    }), Object.defineProperty(this, "onPageLoad", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setCurrentUser(window.currentUser)
                        }
                    }), Object.defineProperty(this, "setCurrentUser", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            var t;
                            const s = null == e.id ? void 0 : e;
                            null != s && this.dataStore.userStore.getOrCreate(s.id, s), this.socketWorker.setUserId(null !== (t = null == s ? void 0 : s.id) && void 0 !== t ? t : null), this.currentUser = s, this.userPreferences.setUser(this.currentUser)
                        }
                    }), e(document).on("turbolinks:load.osu-core", this.onPageLoad), e.subscribe("user:update", this.onCurrentUserUpdate), this.enchant = new o.a(this.turbolinksReload), this.osuAudio = new m.a(this.userPreferences), this.reactTurbolinks = new b.a(this.turbolinksReload), this.userLogin = new w.a(this.captcha), this.dataStore = new x.a, this.userLoginObserver = new _.a, this.windowFocusObserver = new k.a, this.beatmapsetSearchController = new r.a(this.dataStore.beatmapsetSearch), this.socketWorker = new C.a, this.notificationsWorker = new T.a(this.socketWorker), Object(S.p)(this)
                }

                get currentUserOrFail() {
                    if (null == this.currentUser) throw new Error("current user is null");
                    return this.currentUser
                }
            }

            M([S.q], R.prototype, "currentUser", void 0), M([S.h], R.prototype, "currentUserOrFail", null), M([S.f], R.prototype, "setCurrentUser", void 0)
        }).call(this, s("5wds"))
    }, BTwX: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return p
            }));
            var r = s("MtWa"), n = s("Hs9Z"), i = s("/G9H"), a = s("tSlR"), o = s("dTpI"), l = s("cX0L"), c = s("gcbN"),
                u = s("ss8h");
            const d = "beatmap-discussion-editor-toolbar";

            class p extends i.Component {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "ref", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: i.createRef()
                    }), Object.defineProperty(this, "scrollContainer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `editor-toolbar-${Object(l.a)()}`
                    }), Object.defineProperty(this, "scrollTimer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "throttledUpdate", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.throttle(this.updatePosition.bind(this), 100)
                    })
                }

                componentDidMount() {
                    e(window).on(`scroll.${this.eventId}`, this.throttledUpdate), this.updatePosition()
                }

                componentDidUpdate() {
                    this.updatePosition()
                }

                componentWillUnmount() {
                    e(window).off(`.${this.eventId}`), this.scrollContainer && e(this.scrollContainer).off(`.${this.eventId}`), this.throttledUpdate.cancel()
                }

                hide() {
                    const e = this.ref.current;
                    e && (e.style.display = "none")
                }

                render() {
                    return this.context && this.visible() ? i.createElement(r.a, null, i.createElement("div", {
                        ref: this.ref,
                        className: d
                    }, i.createElement(c.a, {format: "bold"}), i.createElement(c.a, {format: "italic"}), i.createElement("div", {className: `${d}__popup-tail`}))) : null
                }

                setScrollContainer(t) {
                    this.scrollContainer && e(this.scrollContainer).off(`.${this.eventId}`), this.scrollContainer = t, e(this.scrollContainer).on(`scroll.${this.eventId}`, this.throttledUpdate)
                }

                updatePosition() {
                    const e = this.ref.current;
                    e && this.context && (this.scrollTimer && window.clearTimeout(this.scrollTimer), this.scrollTimer = window.setTimeout(() => {
                        var t, s, r;
                        if (!this.visible()) return this.hide();
                        for (const e of a.a.positions(this.context, {
                            at: null !== (t = this.context.selection) && void 0 !== t ? t : void 0,
                            unit: "block"
                        })) {
                            if ("embed" === a.c.parent(this.context, e.path).type) return this.hide()
                        }
                        const n = null === (s = this.scrollContainer) || void 0 === s ? void 0 : s.getBoundingClientRect(),
                            i = null !== (r = null == n ? void 0 : n.top) && void 0 !== r ? r : 0,
                            o = null == n ? void 0 : n.bottom,
                            l = window.getSelection().getRangeAt(0).getBoundingClientRect();
                        if (l.top < i || o && l.top > o) return this.hide();
                        e.style.display = "block", e.style.left = `${l.left + (window.pageXOffset - e.offsetWidth) / 2 + l.width / 2}px`, e.style.top = `${l.top - e.clientHeight - 10}px`
                    }, 10))
                }

                visible() {
                    const {selection: e} = this.context;
                    if (!e || !o.b.isFocused(this.context) || a.f.isCollapsed(e) || "" === a.a.string(this.context, e)) return !1;
                    const t = window.getSelection();
                    return null !== (null == t ? void 0 : t.getRangeAt(0))
                }
            }

            Object.defineProperty(p, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: u.a})
        }).call(this, s("5wds"))
    }, BqE9: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("Zp7Q"), n = s("/G9H");

        class i extends n.PureComponent {
            render() {
                return this.props.events.map(e => n.createElement(r.a, {
                    key: e.id,
                    event: e,
                    mode: this.props.mode,
                    users: this.props.users
                }))
            }
        }
    }, BzZm: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return d
            }));
            var r, n, i = s("4QOX"), a = s("f4vq"), o = s("/G9H"), l = (s("I8Ok"), s("FR9d")), c = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, u = {}.hasOwnProperty;
            r = o.createElement, n = a.a.dataStore.uiState;
            var d = function (t) {
                function s() {
                    return this.render = c(this.render, this), this.handleChange = c(this.handleChange, this), s.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) u.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.handleChange = function (t) {
                    return e.publish("comments:sort", {sort: t.target.dataset.value})
                }, s.prototype.render = function () {
                    return r(i.a, null, (e = this, function () {
                        var t;
                        return r(l.a, {
                            currentValue: null != (t = n.comments.loadingSort) ? t : n.comments.currentSort,
                            modifiers: e.props.modifiers,
                            onChange: e.handleChange,
                            values: ["new", "old", "top"]
                        })
                    }));
                    var e
                }, s
            }(o.PureComponent)
        }).call(this, s("5wds"))
    }, C3HX: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return p
            }));
            var r, n, i = s("c1EF"), a = s("1BiD"), o = s("0h6b"), l = s("I8Ok"), c = s("tX/w"), u = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, d = {}.hasOwnProperty;
            n = e.createElement, r = "beatmap-discussion-user-card";
            var p = function (e) {
                function t() {
                    return this.render = u(this.render, this), t.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) d.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(t, e), t.prototype.render = function () {
                    var e, t, s, u, d;
                    return e = null != (u = this.props.additionalClasses) ? u : [], t = null != (d = this.props.hideStripe) && d, s = this.props.user.is_deleted ? l.span : l.a, Object(l.div)({
                        className: Object(c.a)(r, e),
                        style: osu.groupColour(this.props.group)
                    }, Object(l.div)({className: r + "__avatar"}, s({
                        className: r + "__user-link",
                        href: Object(o.a)("users.show", {user: this.props.user.id})
                    }, n(i.a, {
                        user: this.props.user,
                        modifiers: ["full-rounded"]
                    }))), Object(l.div)({className: r + "__user"}, Object(l.div)({className: r + "__user-row"}, s({
                        className: r + "__user-link",
                        href: Object(o.a)("users.show", {user: this.props.user.id})
                    }, Object(l.span)({className: r + "__user-text u-ellipsis-overflow"}, this.props.user.username)), this.props.user.is_bot || this.props.user.is_deleted ? void 0 : Object(l.a)({
                        className: r + "__user-modding-history-link",
                        href: Object(o.a)("users.modding.index", {user: this.props.user.id}),
                        title: osu.trans("beatmap_discussion_posts.item.modding_history_link")
                    }, Object(l.i)({className: "fas fa-align-left"}))), Object(l.div)({className: r + "__user-badge"}, n(a.a, {group: this.props.group}))), t ? void 0 : Object(l.div)({className: r + "__user-stripe"}))
                }, t
            }(e.PureComponent)
        }).call(this, s("/G9H"))
    }, Cfan: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            }));
            var r = s("0h6b"), n = s("lv9K"), i = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            }, a = function (e, t, s, r) {
                return new (s || (s = Promise))((function (n, i) {
                    function a(e) {
                        try {
                            l(r.next(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function o(e) {
                        try {
                            l(r.throw(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function l(e) {
                        var t;
                        e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                            e(t)
                        }))).then(a, o)
                    }

                    l((r = r.apply(e, t || [])).next())
                }))
            };

            class o {
                constructor(e) {
                    Object.defineProperty(this, "id", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "isRevoking", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "name", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "revoked", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "scopes", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "user", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "userId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), this.id = e.id, this.name = e.name, this.scopes = new Set(e.scopes), this.userId = e.user_id, this.user = e.user, Object(n.p)(this)
                }

                revoke() {
                    return a(this, void 0, void 0, (function* () {
                        return this.isRevoking = !0, e.ajax({
                            method: "DELETE",
                            url: Object(r.a)("oauth.authorized-clients.destroy", {authorized_client: this.id})
                        }).then(() => {
                            this.revoked = !0
                        }).always(() => {
                            this.isRevoking = !1
                        })
                    }))
                }
            }

            i([n.q], o.prototype, "isRevoking", void 0), i([n.q], o.prototype, "revoked", void 0), i([n.q], o.prototype, "scopes", void 0), i([n.f], o.prototype, "revoke", null)
        }).call(this, s("5wds"))
    }, Cy5r: function (e, t, s) {
        "use strict";

        function r({allowedBlocks: e = [], allowedInlines: t = []} = {}) {
            e.push("root", "newline"), t.push("text"), this.Parser.prototype.blockMethods.filter(t => !e.includes(t)).forEach(e => {
                this.Parser.prototype.blockTokenizers[e] = () => !0
            }), this.Parser.prototype.inlineMethods.filter(e => !t.includes(e)).forEach(e => {
                this.Parser.prototype.inlineTokenizers[e] = () => !0, this.Parser.prototype.inlineTokenizers[e].locator = () => -1
            })
        }

        s.d(t, "a", (function () {
            return r
        }))
    }, DHbW: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = Object(r.createContext)({})
    }, DXXa: function (e, t, s) {
        "use strict";
        (function (e, r, n) {
            s.d(t, "a", (function () {
                return f
            }));
            var i, a, o = s("0h6b"), l = s("lv9K"), c = s("4QOX"), u = s("f4vq"), d = s("elBb"), p = s("cX0L"),
                m = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, h = {}.hasOwnProperty, b = [].indexOf || function (e) {
                    for (var t = 0, s = this.length; t < s; t++) if (t in this && this[t] === e) return t;
                    return -1
                };
            a = u.a.dataStore.uiState, i = e.createElement;
            var f = function (e) {
                function t(e) {
                    var s, r;
                    this.updateSort = m(this.updateSort, this), this.toggleFollow = m(this.toggleFollow, this), this.saveState = m(this.saveState, this), this.jsonStorageId = m(this.jsonStorageId, this), this.handleCommentUpdated = m(this.handleCommentUpdated, this), this.handleCommentsNew = m(this.handleCommentsNew, this), this.handleCommentsAdded = m(this.handleCommentsAdded, this), this.render = m(this.render, this), this.componentWillUnmount = m(this.componentWillUnmount, this), this.componentDidMount = m(this.componentDidMount, this), t.__super__.constructor.call(this, e), null != e.commentableType && null != e.commentableId && (null != (s = Object(d.c)("json-comments-" + e.commentableType + "-" + e.commentableId, !0)) && (u.a.dataStore.updateWithCommentBundleJson(s), a.initializeWithCommentBundleJson(s)), null != (r = Object(d.c)(this.jsonStorageId())) && a.importCommentsUIState(r)), this.id = "comments-" + Object(p.a)()
                }

                return function (e, t) {
                    for (var s in t) h.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(t, e), t.prototype.SORTS = ["new", "old", "top"], t.prototype.componentDidMount = function () {
                    return r.subscribe("comments:added." + this.id, this.handleCommentsAdded), r.subscribe("comments:new." + this.id, this.handleCommentsNew), r.subscribe("comments:sort." + this.id, this.updateSort), r.subscribe("comments:toggle-follow." + this.id, this.toggleFollow), r.subscribe("comment:updated." + this.id, this.handleCommentUpdated), r(document).on("turbolinks:before-cache." + this.id, this.saveState)
                }, t.prototype.componentWillUnmount = function () {
                    return r.unsubscribe("." + this.id), r(document).off("." + this.id)
                }, t.prototype.render = function () {
                    return i(c.a, null, (e = this, function () {
                        var t;
                        return (t = n.assign({}, e.props.componentProps)).commentableId = e.props.commentableId, t.commentableType = e.props.commentableType, t.user = e.props.user, i(e.props.component, t)
                    }));
                    var e
                }, t.prototype.handleCommentsAdded = function (e, t) {
                    return Object(l.u)((function () {
                        return u.a.dataStore.updateWithCommentBundleJson(t), a.updateFromCommentsAdded(t)
                    }))
                }, t.prototype.handleCommentsNew = function (e, t) {
                    return Object(l.u)((function () {
                        return u.a.dataStore.updateWithCommentBundleJson(t), a.updateFromCommentsNew(t)
                    }))
                }, t.prototype.handleCommentUpdated = function (e, t) {
                    return Object(l.u)((function () {
                        return u.a.dataStore.updateWithCommentBundleJson(t), a.updateFromCommentUpdated(t)
                    }))
                }, t.prototype.jsonStorageId = function () {
                    return "json-comments-manager-state-" + this.props.commentableType + "-" + this.props.commentableId
                }, t.prototype.saveState = function () {
                    if (null != this.props.commentableType && null != this.props.commentableId) return Object(d.d)(this.jsonStorageId(), a.exportCommentsUIState())
                }, t.prototype.toggleFollow = function () {
                    var e;
                    if (e = {
                        follow: {
                            notifiable_type: this.props.commentableType,
                            notifiable_id: this.props.commentableId,
                            subtype: "comment"
                        }
                    }, !a.comments.loadingFollow) return a.comments.loadingFollow = !0, r.ajax(Object(o.a)("follows.store"), {
                        data: e,
                        dataType: "json",
                        method: a.comments.userFollow ? "DELETE" : "POST"
                    }).always((function () {
                        return a.comments.loadingFollow = !1
                    })).done((function () {
                        return a.comments.userFollow = !a.comments.userFollow
                    })).fail((function (e, t) {
                        if ("abort" !== t) return osu.ajaxError(e)
                    }))
                }, t.prototype.updateSort = function (e, t) {
                    var s, n;
                    if (n = t.sort, this.props.commentableType && this.props.commentableId && !(b.call(this.SORTS, n) < 0)) return Object(l.u)((function () {
                        return a.comments.loadingSort = n
                    })), s = {
                        commentable_type: this.props.commentableType,
                        commentable_id: this.props.commentableId,
                        sort: n,
                        parent_id: 0
                    }, r.ajax(Object(o.a)("comments.index"), {data: s, dataType: "json"}).done((function (e) {
                        return u.a.userPreferences.set("comments_sort", n), Object(l.u)((function () {
                            return u.a.dataStore.commentStore.flushStore(), u.a.dataStore.updateWithCommentBundleJson(e), a.initializeWithCommentBundleJson(e)
                        }))
                    })).always((function () {
                        return Object(l.u)((function () {
                            return a.comments.loadingSort = null
                        }))
                    }))
                }, t
            }(e.PureComponent)
        }).call(this, s("/G9H"), s("5wds"), s("Hs9Z"))
    }, DiTM: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("/G9H"), n = s("tX/w");
        const i = e => {
            return `/assets/images/flags/${e.split("").map(e => (e.charCodeAt(0) + 127397).toString(16)).join("-")}.svg`
        };

        function a({country: e, modifiers: t}) {
            return null == e || null == e.code ? null : r.createElement("div", {
                className: Object(n.a)("flag-country", t),
                style: {backgroundImage: `url('${i(e.code)}')`},
                title: e.name
            })
        }
    }, "Do/Y": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("pdUJ"), n = s("0h6b"), i = s("/G9H");

        function a({banner: e}) {
            return null == e ? null : i.createElement("a", {
                className: "profile-tournament-banner",
                href: Object(n.a)("tournaments.show", {tournament: e.tournament_id})
            }, i.createElement(r.a, {className: "profile-tournament-banner__image", src: e.image}))
        }
    }, FR9d: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("f4vq"), n = s("/G9H"), i = s("tX/w");

        class a extends n.PureComponent {
            render() {
                var e;
                const t = this.props.values.map(e => {
                    var t, s;
                    let i = "sort__item sort__item--button";
                    return this.props.currentValue === e && (i += " sort__item--active"), n.createElement("button", {
                        key: e,
                        className: i,
                        "data-value": e,
                        onClick: this.props.onChange
                    }, "rank" === e ? n.createElement("span", null, n.createElement("i", {className: `fas fa-extra-mode-${null !== (s = null === (t = r.a.currentUser) || void 0 === t ? void 0 : t.playmode) && void 0 !== s ? s : "osu"}`}), " ", osu.trans("sort.rank")) : osu.trans(`${this.props.transPrefix}${e}`))
                });
                return n.createElement("div", {className: Object(i.a)("sort", this.props.modifiers)}, n.createElement("div", {className: "sort__items"}, this.props.showTitle && n.createElement("span", {className: "sort__item sort__item--title"}, null !== (e = this.props.title) && void 0 !== e ? e : osu.trans("sort._")), t))
            }
        }

        Object.defineProperty(a, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {showTitle: !0, transPrefix: "sort."}
        })
    }, FiYg: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("lv9K"), n = s("JlDh"), i = s("is6n"), a = s("vZz4"), o = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

            class l {
                constructor(e, t, s) {
                    Object.defineProperty(this, "notificationStore", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e
                    }), Object.defineProperty(this, "contextType", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "currentFilter", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "store", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "typeNamesWithoutNull", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.c.filter(e => !(null == e || this.isExcluded(e)))
                    }), this.currentFilter = void 0 !== s ? s : this.typeNameFromUrl, this.store = t.isWidget ? e.unreadStacks : e.stacks, Object(r.p)(this)
                }

                get stacks() {
                    return this.store.orderedStacksOfType(this.currentFilter).filter(e => e.hasVisibleNotifications && !this.isExcluded(e.objectType))
                }

                get type() {
                    return this.store.getOrCreateType({objectType: this.currentFilter})
                }

                get typeNameFromUrl() {
                    return Object(n.b)(Object(i.b)().get("type"))
                }

                getTotal(e) {
                    return null == e.name ? this.typeNamesWithoutNull.reduce((e, t) => e + this.store.getOrCreateType({objectType: t}).total, 0) : e.total
                }

                getType(e) {
                    return this.store.getOrCreateType({objectType: e})
                }

                loadMore() {
                    var e;
                    null === (e = this.type) || void 0 === e || e.loadMore(this.contextType)
                }

                markCurrentTypeAsRead() {
                    if (null == this.type.name) for (const e of this.typeNamesWithoutNull) this.store.getOrCreateType({objectType: e}).markTypeAsRead(); else this.type.markTypeAsRead()
                }

                navigateTo(t) {
                    if (this.currentFilter = t, 0 === [...this.stacks].length && this.loadMore(), !this.contextType.isWidget) {
                        let s;
                        if (null == t) {
                            const e = new URL(Object(i.a)().href);
                            e.searchParams.delete("type"), s = e.href
                        } else s = Object(a.i)(null, {type: t});
                        e.controller.advanceHistory(s)
                    }
                }

                isExcluded(e) {
                    return this.contextType.excludes.includes(e)
                }
            }

            o([r.q], l.prototype, "currentFilter", void 0), o([r.h], l.prototype, "stacks", null), o([r.h], l.prototype, "type", null), o([r.f], l.prototype, "loadMore", null), o([r.f], l.prototype, "markCurrentTypeAsRead", null), o([r.f], l.prototype, "navigateTo", null)
        }).call(this, s("dMdw"))
    }, FpSo: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("AqrC"), n = s("/G9H"), i = s("ZG74");

        function a(e) {
            const t = n.useContext(i.a), s = n.useContext(i.b),
                a = n.useCallback(() => t.activeKeyDidChange(null), [t]),
                o = n.useCallback(() => t.activeKeyDidChange(s), [t, s]);
            return n.useEffect(() => () => t.activeKeyDidChange(null), [t]), n.createElement(r.a, Object.assign({
                onHide: a,
                onShow: o
            }, e))
        }
    }, G27q: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
            constructor(e) {
                Object.defineProperty(this, "connected", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, HBBY: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return x
            }));
            var n, i, a, o, l, c, u = s("0h6b"), d = s("4QOX"), p = s("f4vq"), m = s("/G9H"), h = s("I8Ok"),
                b = s("tX/w"), f = s("jUJ3"), v = s("phBA"), g = s("tGwB"), y = s("vMSe"), w = s("i41Q"), _ = s("5eFc"),
                O = s("B5xN"), j = s("R9Sp"), E = s("/HbY"), P = s("55pz"), k = s("UBw1"), N = s("c1EF"), S = s("yHuj"),
                T = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, C = {}.hasOwnProperty;
            a = m.createElement, i = {username: osu.trans("users.deleted")}, n = p.a.dataStore.commentableMetaStore, o = p.a.dataStore.commentStore, c = p.a.dataStore.userStore, l = p.a.dataStore.uiState;
            var x = function (t) {
                var s, x, M;

                function R(e) {
                    var t, r;
                    this.toggleClip = T(this.toggleClip, this), this.toggleReplies = T(this.toggleReplies, this), this.closeNewReply = T(this.closeNewReply, this), this.voteToggle = T(this.voteToggle, this), this.toggleNewReply = T(this.toggleNewReply, this), this.restore = T(this.restore, this), this.userFor = T(this.userFor, this), this.parentLink = T(this.parentLink, this), this.loadReplies = T(this.loadReplies, this), this.closeEdit = T(this.closeEdit, this), this.toggleEdit = T(this.toggleEdit, this), this.handleReplyPosted = T(this.handleReplyPosted, this), this.togglePinned = T(this.togglePinned, this), this.delete = T(this.delete, this), this.hasVoted = T(this.hasVoted, this), this.renderToolbar = T(this.renderToolbar, this), this.renderCommentableMeta = T(this.renderCommentableMeta, this), this.renderVoteButton = T(this.renderVoteButton, this), this.renderUsername = T(this.renderUsername, this), this.renderUserAvatar = T(this.renderUserAvatar, this), this.renderToggleClipButton = T(this.renderToggleClipButton, this), this.renderRestore = T(this.renderRestore, this), this.renderReport = T(this.renderReport, this), this.renderReplyButton = T(this.renderReplyButton, this), this.renderReplyBox = T(this.renderReplyBox, this), this.renderRepliesToggle = T(this.renderRepliesToggle, this), this.renderRepliesText = T(this.renderRepliesText, this), this.renderPermalink = T(this.renderPermalink, this), this.renderOwnerBadge = T(this.renderOwnerBadge, this), this.renderEditedBy = T(this.renderEditedBy, this), this.renderEdit = T(this.renderEdit, this), this.renderPin = T(this.renderPin, this), this.renderDeletedBy = T(this.renderDeletedBy, this), this.renderDelete = T(this.renderDelete, this), this.renderComment = T(this.renderComment, this), this.render = T(this.render, this), this.componentDidUpdate = T(this.componentDidUpdate, this), this.componentDidMount = T(this.componentDidMount, this), this.componentWillUnmount = T(this.componentWillUnmount, this), R.__super__.constructor.call(this, e), this.xhr = {}, this.loadMoreRef = m.createRef(), r = !osuCore.windowSize.isMobile && (!this.props.comment.isDeleted && (null != this.props.expandReplies ? this.props.expandReplies : (null != (t = l.getOrderedCommentsByParentId(this.props.comment.id)) ? t.length : void 0) > 0 && this.props.depth < s)), this.state = {
                        clipped: !0,
                        postingVote: !1,
                        editing: !1,
                        showNewReply: !1,
                        expandReplies: r,
                        lines: null
                    }
                }

                return function (e, t) {
                    for (var s in t) C.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(R, t), 7, s = 6, M = document.createElement("div"), x = function (t) {
                    return t.isDeleted ? osu.trans("comments.deleted") : (M.innerHTML = t.messageHtml, e.truncate(M.textContent, {length: 100}))
                }, R.prototype.componentWillUnmount = function () {
                    var e, t, s, r;
                    for (e in s = [], t = this.xhr) C.call(t, e) && (r = t[e], s.push(null != r ? r.abort() : void 0));
                    return s
                }, R.prototype.componentDidMount = function () {
                    var e;
                    return this.setState({lines: Object(f.a)(null != (e = this.props.comment.messageHtml) ? e : "")})
                }, R.prototype.componentDidUpdate = function (e) {
                    var t;
                    if (e.comment.messageHtml !== this.props.comment.messageHtml) return this.setState({lines: Object(f.a)(null != (t = this.props.comment.messageHtml) ? t : "")})
                }, R.prototype.render = function () {
                    return a(d.a, null, (e = this, function () {
                        var t, r, i, c, u, d, p, f, v, g;
                        return e.children = null != (d = l.getOrderedCommentsByParentId(e.props.comment.id)) ? d : [], u = o.comments.get(e.props.comment.parentId), g = e.userFor(e.props.comment), i = n.get(e.props.comment.commentableType, e.props.comment.commentableId), t = null != e.state.lines && e.state.lines.count >= 9, c = null != (p = null != (f = e.props.modifiers) ? f.slice(0) : void 0) ? p : [], 0 === e.props.depth && c.push("top"), r = [], e.props.comment.isDeleted && r.push("deleted"), e.state.clipped && t && r.push("clip"), v = "comment__replies", e.props.depth < s && (v += " comment__replies--indented"), e.state.expandReplies || (v += " comment__replies--hidden"), Object(h.div)({className: Object(b.a)("comment", c)}, e.renderRepliesToggle(), e.renderCommentableMeta(i), e.renderToolbar(), Object(h.div)({
                            className: Object(b.a)("comment__main", r),
                            style: {
                                "--line-height": null != e.state.lines ? e.state.lines.lineHeight + "px" : void 0,
                                "--clip-lines": 7
                            }
                        }, e.props.comment.canHaveVote ? Object(h.div)({className: "comment__float-container comment__float-container--left hidden-xs"}, e.renderVoteButton()) : void 0, e.renderUserAvatar(g), Object(h.div)({className: "comment__container"}, Object(h.div)({className: "comment__row comment__row--header"}, e.renderUsername(g), e.renderOwnerBadge(i), e.props.comment.pinned ? Object(h.span)({className: "comment__row-item  comment__row-item--pinned"}, Object(h.span)({className: "fa fa-thumbtack"}), " ", osu.trans("comments.pinned")) : void 0, null != u ? Object(h.span)({className: "comment__row-item comment__row-item--parent"}, e.parentLink(u)) : void 0, e.props.comment.isDeleted ? Object(h.span)({className: "comment__row-item comment__row-item--deleted"}, osu.trans("comments.deleted")) : void 0), e.state.editing ? Object(h.div)({className: "comment__editor"}, a(y.a, {
                            id: e.props.comment.id,
                            message: e.props.comment.message,
                            modifiers: e.props.modifiers,
                            close: e.closeEdit
                        })) : null != e.props.comment.messageHtml ? a(m.Fragment, null, Object(h.div)({
                            className: "comment__message",
                            dangerouslySetInnerHTML: {__html: e.props.comment.messageHtml}
                        }), t ? e.renderToggleClipButton() : void 0) : void 0, Object(h.div)({className: "comment__row comment__row--footer"}, e.props.comment.canHaveVote ? Object(h.div)({className: "comment__row-item visible-xs"}, e.renderVoteButton(!0)) : void 0, Object(h.div)({className: "comment__row-item comment__row-item--info"}, a(k.a, {
                            dateTime: e.props.comment.createdAt,
                            relative: !0
                        })), e.renderPermalink(), e.renderReplyButton(), e.renderEdit(), e.renderRestore(), e.renderDelete(), e.renderPin(), e.renderReport(), e.renderEditedBy(), e.renderDeletedBy(), e.renderRepliesText()), e.renderReplyBox())), e.props.comment.repliesCount > 0 ? Object(h.div)({className: v}, e.children.map(e.renderComment), a(_.a, {comments: e.children}), a(w.a, {
                            parent: e.props.comment,
                            comments: e.children,
                            total: e.props.comment.repliesCount,
                            modifiers: e.props.modifiers,
                            label: 0 === e.children.length ? osu.trans("comments.load_replies") : void 0,
                            ref: e.loadMoreRef
                        })) : void 0)
                    }));
                    var e
                }, R.prototype.renderComment = function (e) {
                    return (e = o.comments.get(e.id)).isDeleted && !p.a.userPreferences.get("comments_show_deleted") ? null : a(R, {
                        key: e.id,
                        comment: e,
                        depth: this.props.depth + 1,
                        parent: this.props.comment,
                        modifiers: this.props.modifiers,
                        expandReplies: this.props.expandReplies
                    })
                }, R.prototype.renderDelete = function () {
                    if (!this.props.comment.isDeleted && this.props.comment.canDelete) return Object(h.div)({className: "comment__row-item"}, Object(h.button)({
                        type: "button",
                        className: "comment__action",
                        onClick: this.delete
                    }, osu.trans("common.buttons.delete")))
                }, R.prototype.renderDeletedBy = function () {
                    var e;
                    if (this.props.comment.isDeleted && this.props.comment.canModerate) return Object(h.div)({className: "comment__row-item comment__row-item--info"}, a(P.a, {
                        mappings: {
                            timeago: a(k.a, {
                                dateTime: this.props.comment.deletedAt,
                                relative: !0
                            }),
                            user: null != this.props.comment.deletedById ? a(S.a, {user: null != (e = c.get(this.props.comment.deletedById)) ? e : i}) : osu.trans("comments.deleted_by_system")
                        }, pattern: osu.trans("comments.deleted_by")
                    }))
                }, R.prototype.renderPin = function () {
                    if (this.props.comment.canPin) return Object(h.div)({className: "comment__row-item"}, Object(h.button)({
                        type: "button",
                        className: "comment__action",
                        onClick: this.togglePinned
                    }, osu.trans("common.buttons." + (this.props.comment.pinned ? "unpin" : "pin"))))
                }, R.prototype.renderEdit = function () {
                    if (this.props.comment.canEdit) return Object(h.div)({className: "comment__row-item"}, Object(h.button)({
                        type: "button",
                        className: "comment__action " + (this.state.editing ? "comment__action--active" : ""),
                        onClick: this.toggleEdit
                    }, osu.trans("common.buttons.edit")))
                }, R.prototype.renderEditedBy = function () {
                    var e;
                    if (!this.props.comment.isDeleted && this.props.comment.isEdited) return e = c.get(this.props.comment.editedById), Object(h.div)({className: "comment__row-item comment__row-item--info"}, a(P.a, {
                        mappings: {
                            timeago: a(k.a, {
                                dateTime: this.props.comment.editedAt,
                                relative: !0
                            }), user: a(S.a, {user: e})
                        }, pattern: osu.trans("comments.edited")
                    }))
                }, R.prototype.renderOwnerBadge = function (e) {
                    return null == e.owner_id || this.props.comment.userId !== e.owner_id ? null : Object(h.div)({className: "comment__row-item"}, Object(h.div)({className: "comment__owner-badge"}, e.owner_title))
                }, R.prototype.renderPermalink = function () {
                    return Object(h.div)({className: "comment__row-item"}, Object(h.span)({className: "comment__action comment__action--permalink"}, a(g.a, {
                        value: Object(u.a)("comments.show", {comment: this.props.comment.id}),
                        label: osu.trans("common.buttons.permalink"),
                        valueAsUrl: !0
                    })))
                }, R.prototype.renderRepliesText = function () {
                    var e, t;
                    if (0 !== this.props.comment.repliesCount) return this.state.expandReplies || 0 !== this.children.length ? (e = this.toggleReplies, t = osu.transChoice("comments.replies_count", this.props.comment.repliesCount)) : (e = this.loadReplies, t = osu.trans("comments.load_replies")), Object(h.div)({className: "comment__row-item comment__row-item--replies"}, a(j.a, {
                        direction: this.state.expandReplies ? "up" : "down",
                        hasMore: !0,
                        label: t,
                        callback: e,
                        modifiers: ["comment-replies"]
                    }))
                }, R.prototype.renderRepliesToggle = function () {
                    if (0 === this.props.depth && this.children.length > 0) return Object(h.div)({className: "comment__float-container comment__float-container--right"}, Object(h.button)({
                        className: "comment__top-show-replies",
                        type: "button",
                        onClick: this.toggleReplies
                    }, Object(h.span)({className: "fas " + (this.state.expandReplies ? "fa-angle-up" : "fa-angle-down")})))
                }, R.prototype.renderReplyBox = function () {
                    if (this.state.showNewReply) return Object(h.div)({className: "comment__reply-box"}, a(y.a, {
                        close: this.closeNewReply,
                        modifiers: this.props.modifiers,
                        onPosted: this.handleReplyPosted,
                        parent: this.props.comment
                    }))
                }, R.prototype.renderReplyButton = function () {
                    if (!this.props.comment.isDeleted) return Object(h.div)({className: "comment__row-item"}, Object(h.button)({
                        type: "button",
                        className: "comment__action " + (this.state.showNewReply ? "comment__action--active" : ""),
                        onClick: this.toggleNewReply
                    }, osu.trans("common.buttons.reply")))
                }, R.prototype.renderReport = function () {
                    if (this.props.comment.canReport) return Object(h.div)({className: "comment__row-item"}, a(O.a, {
                        className: "comment__action",
                        reportableId: this.props.comment.id,
                        reportableType: "comment",
                        user: this.userFor(this.props.comment)
                    }))
                }, R.prototype.renderRestore = function () {
                    if (this.props.comment.isDeleted && this.props.comment.canRestore) return Object(h.div)({className: "comment__row-item"}, Object(h.button)({
                        type: "button",
                        className: "comment__action",
                        onClick: this.restore
                    }, osu.trans("common.buttons.restore")))
                }, R.prototype.renderToggleClipButton = function () {
                    return Object(h.button)({
                        type: "button",
                        className: "comment__toggle-clip",
                        onClick: this.toggleClip
                    }, this.state.clipped ? osu.trans("common.buttons.read_more") : osu.trans("common.buttons.show_less"))
                }, R.prototype.renderUserAvatar = function (e) {
                    return null != e.id ? Object(h.a)({
                        className: "comment__avatar js-usercard",
                        "data-user-id": e.id,
                        href: Object(u.a)("users.show", {user: e.id})
                    }, a(N.a, {
                        user: e,
                        modifiers: ["full-circle"]
                    })) : Object(h.span)({className: "comment__avatar"}, a(N.a, {user: e, modifiers: ["full-circle"]}))
                }, R.prototype.renderUsername = function (e) {
                    return null != e.id ? Object(h.a)({
                        "data-user-id": e.id,
                        href: Object(u.a)("users.show", {user: e.id}),
                        className: "js-usercard comment__row-item"
                    }, e.username) : Object(h.span)({className: "comment__row-item"}, e.username)
                }, R.prototype.renderVoteButton = function (e) {
                    var t, s, r;
                    return null == e && (e = !1), s = this.hasVoted(), t = Object(b.a)("comment-vote", this.props.modifiers, {
                        disabled: !this.props.comment.canVote,
                        inline: e,
                        on: s,
                        posting: this.state.postingVote
                    }), e || s || (r = Object(h.div)({className: "comment-vote__hover"}, "+1")), Object(h.button)({
                        className: t,
                        type: "button",
                        onClick: this.voteToggle,
                        disabled: this.state.postingVote || !this.props.comment.canVote
                    }, Object(h.span)({className: "comment-vote__text"}, "+" + Object(v.d)(this.props.comment.votesCount, null, {maximumFractionDigits: 1})), this.state.postingVote ? Object(h.span)({className: "comment-vote__spinner"}, a(E.a)) : void 0, r)
                }, R.prototype.renderCommentableMeta = function (e) {
                    var t, s;
                    if (this.props.showCommentableMeta) return e.url ? (t = h.a, s = {
                        href: e.url,
                        className: "comment__link"
                    }) : (t = h.span, s = null), Object(h.div)({className: "comment__commentable-meta"}, null != this.props.comment.commentableType ? Object(h.span)({className: "comment__commentable-meta-type"}, Object(h.span)({className: "comment__commentable-meta-icon fas fa-comment"}), " ", osu.trans("comments.commentable_name." + this.props.comment.commentableType)) : void 0, t(s, e.title))
                }, R.prototype.renderToolbar = function () {
                    if (this.props.showToolbar) return Object(h.div)({className: "comment__toolbar"}, Object(h.div)({className: "sort"}, Object(h.div)({className: "sort__items"}, Object(h.button)({
                        type: "button",
                        className: "sort__item sort__item--button",
                        onClick: this.onShowDeletedToggleClick
                    }, Object(h.span)({className: "sort__item-icon"}, Object(h.span)({className: p.a.userPreferences.get("comments_show_deleted") ? "fas fa-check-square" : "far fa-square"})), osu.trans("common.buttons.show_deleted")))))
                }, R.prototype.hasVoted = function () {
                    return o.userVotes.has(this.props.comment.id)
                }, R.prototype.delete = function () {
                    var e;
                    if (confirm(osu.trans("common.confirmation"))) return null != (e = this.xhr.delete) && e.abort(), this.xhr.delete = r.ajax(Object(u.a)("comments.destroy", {comment: this.props.comment.id}), {method: "DELETE"}).done((function (e) {
                        return r.publish("comment:updated", e)
                    })).fail((function (e, t) {
                        if ("abort" !== t) return osu.ajaxError(e)
                    }))
                }, R.prototype.togglePinned = function () {
                    var e;
                    if (this.props.comment.canPin) return null != (e = this.xhr.pin) && e.abort(), this.xhr.pin = r.ajax(Object(u.a)("comments.pin", {comment: this.props.comment.id}), {method: this.props.comment.pinned ? "DELETE" : "POST"}).done((function (e) {
                        return r.publish("comment:updated", e)
                    })).fail((function (e, t) {
                        if ("abort" !== t) return osu.ajaxError(e)
                    }))
                }, R.prototype.handleReplyPosted = function (e) {
                    if ("reply" === e) return this.setState({expandReplies: !0})
                }, R.prototype.toggleEdit = function () {
                    return this.setState({editing: !this.state.editing})
                }, R.prototype.closeEdit = function () {
                    return this.setState({editing: !1})
                }, R.prototype.loadReplies = function () {
                    var e;
                    return null != (e = this.loadMoreRef.current) && e.load(), this.toggleReplies()
                }, R.prototype.onShowDeletedToggleClick = function () {
                    return p.a.userPreferences.set("comments_show_deleted", !p.a.userPreferences.get("comments_show_deleted"))
                }, R.prototype.parentLink = function (e) {
                    var t, s;
                    return s = {title: x(e)}, this.props.linkParent ? (t = h.a, s.href = Object(u.a)("comments.show", {comment: e.id}), s.className = "comment__link") : t = h.span, t(s, Object(h.span)({className: "fas fa-reply"}), " ", this.userFor(e).username)
                }, R.prototype.userFor = function (e) {
                    var t, s;
                    return null != (s = null != (t = c.get(e.userId)) ? t.toJson() : void 0) ? s : null != e.legacyName ? {username: e.legacyName} : i
                }, R.prototype.restore = function () {
                    var e;
                    return null != (e = this.xhr.restore) && e.abort(), this.xhr.restore = r.ajax(Object(u.a)("comments.restore", {comment: this.props.comment.id}), {method: "POST"}).done((function (e) {
                        return r.publish("comment:updated", e)
                    })).fail((function (e, t) {
                        if ("abort" !== t) return osu.ajaxError(e)
                    }))
                }, R.prototype.toggleNewReply = function () {
                    return this.setState({showNewReply: !this.state.showNewReply})
                }, R.prototype.voteToggle = function (e) {
                    var t, s, n, i, a;
                    if (i = e.target, !p.a.userLogin.showIfGuest(Object(v.c)(i))) return this.setState({postingVote: !0}), this.hasVoted() ? (t = "DELETE", n = "removeUserVote") : (t = "POST", n = "addUserVote"), null != (s = this.xhr.vote) && s.abort(), this.xhr.vote = r.ajax(Object(u.a)("comments.vote", {comment: this.props.comment.id}), {method: t}).always((a = this, function () {
                        return a.setState({postingVote: !1})
                    })).done(function (e) {
                        return function (t) {
                            return r.publish("comment:updated", t), o[n](e.props.comment)
                        }
                    }(this)).fail((function (e, t) {
                        if ("abort" !== t) return 401 === e.status ? r(i).trigger("ajax:error", [e, t]) : osu.ajaxError(e)
                    }))
                }, R.prototype.closeNewReply = function () {
                    return this.setState({showNewReply: !1})
                }, R.prototype.toggleReplies = function () {
                    return this.setState({expandReplies: !this.state.expandReplies})
                }, R.prototype.toggleClip = function () {
                    return this.setState({clipped: !this.state.clipped})
                }, R
            }(m.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, HUtF: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            }));
            var r = s("/G9H"), n = s("I8Ok"), i = s("WKXC"), a = s("+bOe"), o = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, l = {}.hasOwnProperty, c = function (t) {
                function s() {
                    this.renderPortalContent = o(this.renderPortalContent, this), this.render = o(this.render, this), this.hideModal = o(this.hideModal, this), this.handleMouseUp = o(this.handleMouseUp, this), this.handleMouseDown = o(this.handleMouseDown, this), this.handleEsc = o(this.handleEsc, this), this.componentWillUnmount = o(this.componentWillUnmount, this), this.componentDidUpdate = o(this.componentDidUpdate, this), this.componentDidMount = o(this.componentDidMount, this), this.ref = Object(r.createRef)()
                }

                return function (e, t) {
                    for (var s in t) l.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.isOpen = function () {
                    return document.body.classList.contains("js-react-modal---is-open")
                }, s.prototype.close = function () {
                    return document.body.classList.remove("js-react-modal---is-open"), Object(a.c)(!1, .5)
                }, s.prototype.componentDidMount = function () {
                    var t;
                    return document.body.appendChild(this.portal), document.addEventListener("keydown", this.handleEsc), e(document).on("turbolinks:before-cache.modal", (t = this, function () {
                        return t.close(), document.body.removeChild(t.portal)
                    })), this.props.visible ? this.open() : this.close()
                }, s.prototype.componentDidUpdate = function (e) {
                    if (e.visible !== this.props.visible) return this.props.visible ? this.open() : this.close()
                }, s.prototype.componentWillUnmount = function () {
                    return this.close(), document.removeEventListener("keydown", this.handleEsc), e(document).off(".modal")
                }, s.prototype.handleEsc = function (e) {
                    var t;
                    if (27 === e.keyCode) return "function" == typeof (t = this.props).onClose ? t.onClose() : void 0
                }, s.prototype.handleMouseDown = function (e) {
                    return this.clickStartTarget = e.target
                }, s.prototype.handleMouseUp = function (e) {
                    return this.clickEndTarget = e.target
                }, s.prototype.hideModal = function (e) {
                    var t;
                    if (0 === e.button && e.target === this.ref.current && this.clickEndTarget === this.clickStartTarget) return "function" == typeof (t = this.props).onClose ? t.onClose() : void 0
                }, s.prototype.open = function () {
                    return document.body.classList.add("js-react-modal---is-open"), Object(a.c)(!0, .5)
                }, s.prototype.render = function () {
                    return null == this.portal && (this.portal = document.createElement("div")), Object(i.createPortal)(this.renderPortalContent(), this.portal)
                }, s.prototype.renderPortalContent = function () {
                    return Object(n.div)({
                        className: "js-react-modal",
                        onClick: this.hideModal,
                        onMouseDown: this.handleMouseDown,
                        onMouseUp: this.handleMouseUp,
                        ref: this.ref
                    }, this.props.children)
                }, s
            }(r.PureComponent)
        }).call(this, s("5wds"))
    }, I7Md: function (e, t, s) {
        "use strict";
        (function (e) {
            var r = s("WLnA"), n = s("0h6b"), i = s("lv9K"), a = s("tz7b"), o = s("8gxX"), l = s("uW+8"),
                c = function (e, t, s, r) {
                    var n, i = arguments.length,
                        a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };
            let u = class {
                constructor(t) {
                    Object.defineProperty(this, "socketWorker", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "waitingVerification", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "firstLoadedAt", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "retryDelay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new o.a
                    }), Object.defineProperty(this, "timeout", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "xhrLoadingState", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "loadMore", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.xhrLoadingState.loadMore || (window.clearTimeout(this.timeout.loadMore), this.xhrLoadingState.loadMore = !0, this.xhr.loadMore = e.ajax({
                                dataType: "json",
                                url: Object(n.a)("notifications.index", {unread: 1})
                            }).always(() => {
                                this.xhrLoadingState.loadMore = !1
                            }).done(Object(i.f)(e => {
                                this.waitingVerification = !1, this.loadBundle(e), this.retryDelay.reset()
                            })).fail(Object(i.f)(e => {
                                null == e.responseJSON || "verification" !== e.responseJSON.error ? 401 !== e.status && this.delayedRetryInitialLoadMore() : this.waitingVerification = !0
                            })))
                        }
                    }), Object(i.r)(this.socketWorker, "connectionStatus", e => {
                        "connected" === e.newValue && this.loadMore()
                    }, !0), e.subscribe("user-verification:success.notifications-worker", () => {
                        this.loadMore()
                    }), Object(i.p)(this)
                }

                get hasData() {
                    return null != this.firstLoadedAt
                }

                handleDispatchAction(e) {
                    if (!(e instanceof a.a)) return;
                    const t = e.message;
                    if ((e => "delete" === e.event)(t)) {
                        const e = new Date(t.data.timestamp);
                        null != this.firstLoadedAt && e > this.firstLoadedAt && Object(r.a)(l.a.fromJson(t))
                    } else if ((e => "new" === e.event)(t)) Object(r.a)(new l.c(t.data)); else if ((e => "read" === e.event)(t)) {
                        const e = new Date(t.data.timestamp);
                        null != this.firstLoadedAt && e > this.firstLoadedAt && Object(r.a)(l.d.fromJson(t))
                    }
                }

                delayedRetryInitialLoadMore() {
                    this.timeout.loadMore = window.setTimeout(this.loadMore, this.retryDelay.get())
                }

                loadBundle(e) {
                    Object(r.a)(new l.b(e, {isWidget: !0})), null == this.firstLoadedAt && (this.firstLoadedAt = new Date(e.timestamp))
                }
            };
            c([i.q], u.prototype, "waitingVerification", void 0), c([i.q], u.prototype, "firstLoadedAt", void 0), c([i.h], u.prototype, "hasData", null), c([i.f], u.prototype, "loadBundle", null), u = c([r.b], u), t.a = u
        }).call(this, s("5wds"))
    }, IWDN: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");

        function n(e) {
            return r.createElement("h3", {className: "title title--page-extra-small"}, osu.trans(e.titleKey), null != e.count && r.createElement("span", {className: "title__count"}, osu.formatNumber(e.count)))
        }
    }, IfWv: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("0h6b"), n = s("lv9K"), i = s("Cfan"), a = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            }, o = function (e, t, s, r) {
                return new (s || (s = Promise))((function (n, i) {
                    function a(e) {
                        try {
                            l(r.next(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function o(e) {
                        try {
                            l(r.throw(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function l(e) {
                        var t;
                        e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                            e(t)
                        }))).then(a, o)
                    }

                    l((r = r.apply(e, t || [])).next())
                }))
            };

            class l extends i.a {
                constructor(e) {
                    super(e), Object.defineProperty(this, "isResetting", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "isUpdating", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "redirect", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "secret", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), this.redirect = e.redirect, this.secret = e.secret, Object(n.p)(this)
                }

                delete() {
                    return o(this, void 0, void 0, (function* () {
                        return this.isRevoking = !0, e.ajax({
                            method: "DELETE",
                            url: Object(r.a)("oauth.clients.destroy", {client: this.id})
                        }).then(() => {
                            this.revoked = !0
                        }).always(() => {
                            this.isRevoking = !1
                        })
                    }))
                }

                resetSecret() {
                    return o(this, void 0, void 0, (function* () {
                        return this.isResetting = !0, e.ajax({
                            method: "POST",
                            url: Object(r.a)("oauth.clients.reset-secret", {client: this.id})
                        }).then(e => {
                            this.updateFromJson(e)
                        }).always(() => {
                            this.isResetting = !1
                        })
                    }))
                }

                updateFromJson(e) {
                    this.id = e.id, this.name = e.name, this.scopes = new Set(e.scopes), this.userId = e.user_id, this.user = e.user, this.redirect = e.redirect, this.secret = e.secret
                }

                updateWith(t) {
                    return o(this, void 0, void 0, (function* () {
                        const {redirect: s} = t;
                        return this.isUpdating = !0, e.ajax({
                            data: {redirect: s},
                            method: "PUT",
                            url: Object(r.a)("oauth.clients.update", {client: this.id})
                        }).then(e => {
                            this.updateFromJson(e)
                        }).always(() => {
                            this.isUpdating = !1
                        })
                    }))
                }
            }

            a([n.q], l.prototype, "isResetting", void 0), a([n.q], l.prototype, "isUpdating", void 0), a([n.f], l.prototype, "delete", null), a([n.f], l.prototype, "resetSecret", null), a([n.f], l.prototype, "updateFromJson", null), a([n.f], l.prototype, "updateWith", null)
        }).call(this, s("5wds"))
    }, Irky: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");

        class n extends r.PureComponent {
            render() {
                return 0 === this.props.badges.length ? null : r.createElement("div", {className: "profile-badges"}, this.props.badges.map(e => {
                    const t = {
                        className: "profile-badges__badge",
                        key: e.image_url,
                        style: {backgroundImage: osu.urlPresence(e.image_url)},
                        title: e.description
                    };
                    return osu.present(e.url) ? r.createElement("a", Object.assign({href: e.url}, t)) : r.createElement("span", Object.assign({}, t))
                }))
            }
        }

        Object.defineProperty(n, "defaultProps", {enumerable: !0, configurable: !0, writable: !0, value: {badges: []}})
    }, IwO5: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        class i extends r.PureComponent {
            constructor() {
                super(...arguments), Object.defineProperty(this, "bn", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: "circular-progress"
                })
            }

            render() {
                const e = this.bn, t = Math.min(1, this.props.current / this.props.max);
                if (this.props.onlyShowAsWarning && t < .75) return null;
                const s = this.props.ignoreProgress ? void 0 : {transform: `rotate(${t}turn)`};
                return r.createElement("div", {
                    className: Object(n.a)(e, {
                        over: 1 === t,
                        over50: t > .5,
                        warn: t >= .75 && t < 1,
                        [this.props.theme]: osu.present(this.props.theme)
                    }), title: this.props.tooltip || `${this.props.current} / ${this.props.max}`
                }, r.createElement("div", {className: `${e}__label`}, this.props.max - this.props.current), r.createElement("div", {className: `${e}__slice`}, r.createElement("div", {
                    className: `${e}__circle`,
                    style: s
                }), r.createElement("div", {className: `${e}__circle ${e}__circle--fill`})))
            }
        }

        Object.defineProperty(i, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {ignoreProgress: !1, onlyShowAsWarning: !1, theme: ""}
        })
    }, JlDh: function (e, t, s) {
        "use strict";
        s.d(t, "c", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        })), s.d(t, "a", (function () {
            return o
        }));
        var r = s("lv9K"), n = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        const i = [null, "user", "beatmapset", "forum_topic", "news_post", "build", "channel"];

        function a(e) {
            const t = e;
            return i.indexOf(t) > -1 ? t : i[0]
        }

        class o {
            constructor(e, t) {
                Object.defineProperty(this, "name", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "resolver", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: t
                }), Object.defineProperty(this, "cursor", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "isDeleting", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isLoading", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isMarkingAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "stacks", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object.defineProperty(this, "total", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), Object(r.p)(this)
            }

            get hasMore() {
                return null !== this.cursor && this.stackNotificationCount < this.total
            }

            get hasVisibleNotifications() {
                return this.total > 0 && this.stacks.size > 0
            }

            get identity() {
                return {objectType: this.name}
            }

            get isEmpty() {
                return this.total <= 0
            }

            get stackNotificationCount() {
                return [...this.stacks.values()].reduce((e, t) => e + t.total, 0)
            }

            static fromJson(e, t) {
                const s = new o(e.name, t);
                return s.updateWithJson(e), s
            }

            delete() {
                this.resolver.delete(this)
            }

            loadMore(e) {
                null !== this.cursor && (this.isLoading = !0, this.resolver.loadMore(this.identity, e, this.cursor).always(Object(r.f)(() => {
                    this.isLoading = !1
                })))
            }

            markTypeAsRead() {
                this.resolver.queueMarkAsRead(this)
            }

            removeStack(e) {
                const t = this.stacks.delete(e.id);
                return t && (this.total -= e.total), t
            }

            updateWithJson(e) {
                this.cursor = e.cursor, this.total = e.total
            }
        }

        n([r.q], o.prototype, "cursor", void 0), n([r.q], o.prototype, "isDeleting", void 0), n([r.q], o.prototype, "isLoading", void 0), n([r.q], o.prototype, "isMarkingAsRead", void 0), n([r.q], o.prototype, "stacks", void 0), n([r.q], o.prototype, "total", void 0), n([r.h], o.prototype, "hasMore", null), n([r.h], o.prototype, "hasVisibleNotifications", null), n([r.h], o.prototype, "isEmpty", null), n([r.h], o.prototype, "stackNotificationCount", null), n([r.f], o.prototype, "delete", null), n([r.f], o.prototype, "loadMore", null), n([r.f], o.prototype, "markTypeAsRead", null), n([r.f], o.prototype, "removeStack", null), n([r.f], o.prototype, "updateWithJson", null)
    }, JsCm: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("Hs9Z");
            const n = e => {
                var t;
                return null !== (t = e.clientX) && void 0 !== t ? t : e.touches[0].clientX
            };
            let i = null;

            class a {
                constructor(t) {
                    Object.defineProperty(this, "active", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !0
                    }), Object.defineProperty(this, "bar", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "endCallback", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "moveCallback", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "percentage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: 0
                    }), Object.defineProperty(this, "end", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.active = !1, e(document).off("mousemove touchmove", this.onMove), e(document).off("mouseup touchend", this.end), e(window).off("blur", this.end), e(document).off("turbolinks:before-cache", this.end), null != this.endCallback && this.endCallback(this), this.bar.style.removeProperty("--bar"), this.bar.dataset.audioDragging = "0", i = null
                        }
                    }), Object.defineProperty(this, "getPercentage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => this.percentage
                    }), Object.defineProperty(this, "move", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (!this.active) return;
                            const t = this.bar.getBoundingClientRect(), s = e - t.left, n = t.width;
                            this.percentage = Object(r.clamp)(s / n, 0, 1), null != this.moveCallback && this.moveCallback(this), this.bar.style.setProperty("--bar", this.percentage.toString())
                        }
                    }), Object.defineProperty(this, "onMove", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = n(e);
                            requestAnimationFrame(() => {
                                this.move(t)
                            })
                        }
                    }), this.endCallback = t.endCallback, this.moveCallback = t.moveCallback, this.bar = t.bar, this.bar.dataset.audioDragging = "1", this.move(n(t.initialEvent)), e(document).on("mousemove touchmove", this.onMove), e(document).on("mouseup touchend", this.end), e(window).on("blur", this.end), e(document).on("turbolinks:before-cache", this.end)
                }

                static start(e) {
                    if (0 === e.initialEvent.which || 1 === e.initialEvent.which) return null != i && i.end(), i = new a(e)
                }
            }

            Object.defineProperty(a, "startEvents", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: "mousedown touchstart"
            })
        }).call(this, s("5wds"))
    }, KMrT: function (e, t, s) {
        "use strict";
        var r = s("KUml"), n = s("/G9H"), i = s("R9Sp"), a = s("2etm"), o = s("Hs9Z"), l = s("y2EG");

        function c(e) {
            return "beatmap_owner_change" === e.name
        }

        l.a;

        function u(e, t = !1) {
            const s = {content: e.details.content, title: e.title, username: e.details.username};
            c(e) && (s.beatmap = e.details.version), "beatmapset_discussion_review_new" === e.name && null != e.details.embeds && o.merge(s, {
                praises: e.details.embeds.praises,
                problems: e.details.embeds.problems,
                suggestions: e.details.embeds.suggestions
            });
            let r = `notifications.item.${e.displayType}.${e.category}`;
            "channel" === e.objectType && (r += `.${e.details.type}`), r += `.${e.name}`, t && (r += "_compact");
            const n = `${r}_empty`;
            return null == e.details.content && osu.transExists(n) && (r = n), osu.trans(r, s)
        }

        function d(e) {
            if ("channel" === e.objectType) {
                const t = {title: e.title, username: e.details.username},
                    s = `notifications.item.${e.objectType}.${e.category}.${e.details.type}.${e.name}_group`;
                return osu.trans(s, t)
            }
            if ("user_achievement_unlock" === e.name) {
                const t = {username: e.details.username};
                return osu.trans(`notifications.item.${e.displayType}.${e.category}.${e.name}_group`, t)
            }
            if ("user_beatmapset_new" === e.category) {
                const t = {username: e.details.username};
                return osu.trans(`notifications.item.${e.displayType}.${e.category}.${e.category}_group`, t)
            }
            return e.title
        }

        var p = s("0h6b");

        function m(e) {
            if (c(e)) return Object(p.a)("beatmapsets.discussion", {
                beatmap: "-",
                beatmapset: e.objectId,
                mode: "events"
            });
            if ("comment_new" === e.name || "comment_reply" === e.name) switch (e.objectType) {
                case"beatmapset":
                    return Object(p.a)("beatmapsets.show", {beatmapset: e.objectId});
                case"build":
                    return Object(p.a)("changelog.show", {changelog: e.objectId, key: "id"});
                case"news_post":
                    return Object(p.a)("news.show", {key: "id", news: e.objectId})
            } else {
                if ("user_achievement_unlock" === e.name) return b(e);
                if ("user_beatmapset_new" === e.category) return `${Object(p.a)("users.show", {user: e.objectId})}#beatmaps`
            }
            switch (e.objectType) {
                case"beatmapset":
                    return Object(p.a)("beatmapsets.discussion", {beatmapset: e.objectId});
                case"channel":
                    return null != e.details.channel_id ? Object(p.a)("chat.index", {channel_id: e.details.channelId}) : Object(p.a)("chat.index", {sendto: e.sourceUserId});
                case"forum_topic":
                    return Object(p.a)("forum.topics.show", {start: "unread", topic: e.objectId})
            }
        }

        function h(e) {
            if (c(e)) return Object(p.a)("beatmapsets.discussion", {
                beatmap: e.details.beatmapId,
                beatmapset: e.objectId
            });
            switch (e.name) {
                case"beatmapset_discussion_lock":
                case"beatmapset_discussion_unlock":
                case"beatmapset_disqualify":
                case"beatmapset_love":
                case"beatmapset_nominate":
                case"beatmapset_qualify":
                case"beatmapset_remove_from_loved":
                case"beatmapset_reset_nominations":
                    return Object(p.a)("beatmapsets.discussion", {beatmapset: e.objectId});
                case"beatmapset_discussion_post_new":
                case"beatmapset_discussion_qualified_problem":
                case"beatmapset_discussion_review_new":
                    return BeatmapDiscussionHelper.url({
                        beatmapId: e.details.beatmapId,
                        beatmapsetId: e.objectId,
                        discussionId: e.details.discussionId
                    });
                case"beatmapset_rank":
                    return Object(p.a)("beatmapsets.show", {beatmapset: e.objectId});
                case"channel_announcement":
                    return Object(p.a)("chat.index", {channel_id: e.details.channelId});
                case"channel_message":
                    return Object(p.a)("chat.index", {sendto: e.sourceUserId});
                case"comment_new":
                case"comment_reply":
                    return Object(p.a)("comments.show", {comment: e.details.commentId});
                case"forum_topic_reply":
                    return Object(p.a)("forum.posts.show", {post: e.details.postId});
                case"user_achievement_unlock":
                    return b(e);
                case"user_beatmapset_new":
                case"user_beatmapset_revive":
                    return Object(p.a)("beatmapsets.show", {beatmapset: e.details.beatmapsetId})
            }
        }

        function b(e) {
            var t;
            const s = {
                mode: null !== (t = e.details.achievementMode) && void 0 !== t ? t : void 0,
                user: e.details.userId
            };
            return `${Object(p.a)("users.show", s)}#medals`
        }

        var f = s("h/Ip"), v = s("8Xmz"), g = s("9zVE"), y = s("mjdM");
        let w = class extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "handleDelete", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.deleteItem(this.props.item)
                    }
                }), Object.defineProperty(this, "handleMarkAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.markAsRead(this.props.item)
                    }
                })
            }

            render() {
                return n.createElement(y.a, {
                    delete: this.handleDelete,
                    icons: a.c[this.props.item.name || ""],
                    item: this.props.item,
                    markRead: this.handleMarkAsRead,
                    message: u(this.props.item, !0),
                    modifiers: ["compact"],
                    url: h(this.props.item),
                    withCategory: !1,
                    withCoverImage: "user_achievement_unlock" === this.props.item.name || "user_beatmapset_new" === this.props.item.category
                })
            }
        };
        var _ = w = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], w), O = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let j = class extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "state", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {expanded: !1}
                }), Object.defineProperty(this, "handleDelete", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.delete()
                    }
                }), Object.defineProperty(this, "handleMarkAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.markStackAsRead()
                    }
                }), Object.defineProperty(this, "handleShowLess", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({expanded: !1})
                    }
                }), Object.defineProperty(this, "handleShowMore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.loadMore(this.context)
                    }
                }), Object.defineProperty(this, "renderItem", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => n.createElement("div", {
                        key: e.id,
                        className: "notification-popup-item-group__item"
                    }, n.createElement(_, {item: e, stack: this.props.stack}))
                }), Object.defineProperty(this, "toggleExpand", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({expanded: !this.state.expanded})
                    }
                })
            }

            render() {
                const e = this.props.stack.first;
                return null == e ? null : n.createElement("div", {className: "notification-popup-item-group"}, n.createElement(y.a, {
                    canMarkAsRead: this.props.stack.canMarkAsRead,
                    delete: this.handleDelete,
                    expandButton: this.renderExpandButton(),
                    icons: a.a[e.category],
                    isDeleting: this.props.stack.isDeleting,
                    isMarkingAsRead: this.props.stack.isMarkingAsRead,
                    item: e,
                    markRead: this.handleMarkAsRead,
                    message: d(e),
                    modifiers: ["group"],
                    url: m(e),
                    withCategory: !0,
                    withCoverImage: !0
                }), this.renderItems())
            }

            renderExpandButton() {
                const e = this.props.stack.total,
                    t = this.context.isWidget ? "common.count.update" : "common.count.notifications";
                return n.createElement("button", {
                    className: "show-more-link show-more-link--notification-group",
                    onClick: this.toggleExpand,
                    type: "button"
                }, n.createElement("span", {className: "show-more-link__label"}, n.createElement("span", {className: "show-more-link__label-text"}, osu.transChoice(t, e)), n.createElement("span", {className: "show-more-link__label-icon"}, n.createElement("span", {className: `fas fa-angle-${this.state.expanded ? "up" : "down"}`}))))
            }

            renderItems() {
                return this.state.expanded ? n.createElement("div", {className: "notification-popup-item-group__items"}, this.props.stack.orderedNotifications.map(this.renderItem), n.createElement("div", {className: "notification-popup-item-group__show-more"}, n.createElement("div", {className: "notification-popup-item-group__expand"}, this.renderShowMore()), n.createElement("div", {className: "notification-popup-item-group__collapse"}, this.renderShowLess(), this.props.stack.canMarkAsRead && n.createElement(g.a, {
                    isMarkingAsRead: this.props.stack.isMarkingAsRead,
                    onMarkAsRead: this.handleMarkAsRead
                }), !this.context.isWidget && n.createElement(v.a, {
                    isDeleting: this.props.stack.isDeleting,
                    onDelete: this.handleDelete
                })))) : null
            }

            renderShowLess() {
                return n.createElement(i.a, {
                    callback: this.handleShowLess,
                    direction: "up",
                    hasMore: !0,
                    label: osu.trans("common.buttons.show_less"),
                    modifiers: ["notification-group"]
                })
            }

            renderShowMore() {
                const e = this.props.stack;
                return e.hasMore ? n.createElement(i.a, {
                    callback: this.handleShowMore,
                    hasMore: null != e.cursor,
                    loading: e.isLoading,
                    modifiers: ["notification-group"]
                }) : null
            }
        };
        Object.defineProperty(j, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: f.a});
        var E = j = O([r.b], j);
        let P = class extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "handleDelete", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.deleteItem(this.props.stack.first)
                    }
                }), Object.defineProperty(this, "handleMarkAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.props.stack.markAsRead(this.props.stack.first)
                    }
                })
            }

            render() {
                const e = this.props.stack.first;
                return null == e ? null : n.createElement(y.a, {
                    delete: this.handleDelete,
                    icons: a.b[e.name || ""],
                    isDeleting: e.isDeleting,
                    isMarkingAsRead: e.isMarkingAsRead,
                    item: e,
                    markRead: this.handleMarkAsRead,
                    message: u(e),
                    modifiers: ["one"],
                    url: h(e),
                    withCategory: !0,
                    withCoverImage: !0
                })
            }
        };
        var k = P = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], P);
        let N = class extends n.Component {
            render() {
                if (!this.props.stack.hasVisibleNotifications) return null;
                const e = this.props.stack.isSingle ? k : E;
                return n.createElement(e, {key: this.props.stack.id, stack: this.props.stack})
            }
        };
        N = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], N);
        t.a = N
    }, Kexm: function (e, t, s) {
        "use strict";
        var r = s("lv9K"), n = s("KUml"), i = s("f4vq"), a = s("/G9H"), o = s("/jJF"), l = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let c = class extends a.Component {
            constructor(e) {
                super(e), Object.defineProperty(this, "onUpdateCallbacks", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "onClick", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        const e = this.isPinned ? "0" : "1";
                        this.onUpdateCallbacks = {
                            done: this.props.onUpdate,
                            fail: Object(o.c)(this.onClick)
                        }, i.a.scorePins.apiPin(this.props.score, !this.isPinned).done(() => {
                            var t, s;
                            osu.popup(osu.trans(`users.show.extra.top_ranks.pin.to_${e}_done`), "info"), null === (s = null === (t = this.onUpdateCallbacks) || void 0 === t ? void 0 : t.done) || void 0 === s || s.call(t)
                        }).fail((e, t) => {
                            var s;
                            return null === (s = this.onUpdateCallbacks) || void 0 === s ? void 0 : s.fail(e, t)
                        })
                    }
                }), Object(r.p)(this)
            }

            get isPinned() {
                return i.a.scorePins.isPinned(this.props.score)
            }

            get label() {
                const e = this.isPinned ? "0" : "1";
                return osu.trans(`users.show.extra.top_ranks.pin.to_${e}`)
            }

            componentWillUnmount() {
                this.onUpdateCallbacks = null
            }

            render() {
                return a.createElement("button", {
                    className: this.props.className,
                    onClick: this.onClick,
                    type: "button"
                }, this.label)
            }
        };
        l([r.h], c.prototype, "isPinned", null), l([r.h], c.prototype, "label", null), c = l([n.b], c), t.a = c
    }, LOag: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return m
            }));
            var r, n, i = s("55pz"), a = s("UBw1"), o = s("yHuj"), l = s("0h6b"), c = s("/G9H"), u = s("I8Ok"),
                d = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, p = {}.hasOwnProperty;
            n = c.createElement, r = "beatmapset-mapping";
            var m = function (t) {
                function s() {
                    return this.renderDate = d(this.renderDate, this), this.render = d(this.render, this), s.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) p.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.render = function () {
                    var e, t, s, a;
                    return a = {
                        id: this.props.beatmapset.user_id,
                        username: null != (e = this.props.beatmapset.creator) ? e : osu.trans("users.deleted"),
                        avatar_url: null != (t = null != (s = this.props.user) ? s : this.props.beatmapset.user) ? t.avatar_url : void 0
                    }, Object(u.div)({className: r}, null != a.id ? Object(u.a)({
                        href: Object(l.a)("users.show", {user: a.id}),
                        className: "avatar avatar--beatmapset",
                        style: {backgroundImage: osu.urlPresence(a.avatar_url)}
                    }) : Object(u.span)({className: "avatar avatar--beatmapset avatar--guest"}), Object(u.div)({className: r + "__content"}, Object(u.div)({className: r + "__mapper"}, n(i.a, {
                        pattern: osu.trans("beatmapsets.show.details.mapped_by"),
                        mappings: {mapper: n(o.a, {className: r + "__user", user: a})}
                    })), this.renderDate("submitted", "submitted_date"), this.props.beatmapset.ranked > 0 ? this.renderDate(this.props.beatmapset.status, "ranked_date") : this.renderDate("updated", "last_updated")))
                }, s.prototype.renderDate = function (t, s) {
                    return Object(u.div)(null, n(i.a, {
                        pattern: osu.trans("beatmapsets.show.details_date." + t),
                        mappings: {
                            timeago: Object(u.strong)(null, n(a.a, {
                                dateTime: this.props.beatmapset[s],
                                relative: Math.abs(e().diff(e(this.props.beatmapset[s]), "weeks")) < 4
                            }))
                        }
                    }))
                }, s
            }(c.PureComponent)
        }).call(this, s("7EfK"))
    }, LPNJ: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            var r = s("HUtF");

            function n() {
                return r.a.isOpen() || e("#overlay").is(":visible") || document.body.classList.contains("modal-open")
            }
        }).call(this, s("5wds"))
    }, "LXy+": function (e, t, s) {
        "use strict";

        function r(e) {
            return e.legacy_mode
        }

        s.d(t, "a", (function () {
            return r
        }))
    }, Ma8u: function (e, t, s) {
        "use strict";
        s.d(t, "b", (function () {
            return i
        })), s.d(t, "a", (function () {
            return a
        }));
        var r = s("Hs9Z");

        function n() {
            const e = document.querySelector(".js-loading-overlay");
            if (e instanceof HTMLElement) return e
        }

        const i = Object(r.debounce)((function () {
            const e = n();
            null != e && e.classList.add("loading-overlay--visible")
        }), 5e3, {maxWait: 5e3});

        function a() {
            i.cancel();
            const e = n();
            null != e && e.classList.remove("loading-overlay--visible")
        }
    }, MtWa: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("/G9H"), n = s("WKXC"), i = s("cX0L");

            class a extends r.PureComponent {
                constructor(t) {
                    super(t), Object.defineProperty(this, "container", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `portal-${Object(i.a)()}`
                    }), Object.defineProperty(this, "addPortal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => document.body.appendChild(this.container)
                    }), Object.defineProperty(this, "componentWillUnmount", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.removePortal(), e(document).off(`turbolinks:before-cache.${this.eventId}`)
                        }
                    }), Object.defineProperty(this, "removePortal", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.container.parentElement === document.body && document.body.removeChild(this.container)
                        }
                    }), Object.defineProperty(this, "render", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => Object(n.createPortal)(this.props.children, this.container)
                    }), this.container = document.createElement("div")
                }

                componentDidMount() {
                    this.addPortal(), e(document).on(`turbolinks:before-cache.${this.eventId}`, this.removePortal)
                }
            }
        }).call(this, s("5wds"))
    }, N05Q: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("0h6b");

        function n(e) {
            return [{
                active: "home" === e,
                title: osu.trans("home.user.title"),
                url: Object(r.a)("home")
            }, {
                active: "friends.index" === e,
                title: osu.trans("friends.title_compact"),
                url: Object(r.a)("friends.index")
            }, {
                active: "follows.index" === e,
                title: osu.trans("follows.index.title_compact"),
                url: Object(r.a)("follows.index", {subtype: "forum_topic"})
            }, {
                active: "account.edit" === e,
                title: osu.trans("accounts.edit.title_compact"),
                url: Object(r.a)("account.edit")
            }]
        }
    }, NfKI: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("/G9H"), n = s("ZG74"), i = s("tX/w"), a = s("6WEK");

        class o extends r.PureComponent {
            constructor() {
                super(...arguments), Object.defineProperty(this, "activeKeyDidChange", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: n.c.bind(this)
                }), Object.defineProperty(this, "state", {enumerable: !0, configurable: !0, writable: !0, value: {}})
            }

            render() {
                const e = null != this.state.activeKey ? ["menu-active"] : [];
                return e.push(this.props.viewMode), r.createElement(n.a.Provider, {value: {activeKeyDidChange: this.activeKeyDidChange}}, r.createElement("div", {className: Object(i.a)("user-cards", e)}, this.props.users.map(e => {
                    const t = this.state.activeKey === e.id;
                    return r.createElement(n.b.Provider, {key: e.id, value: e.id}, r.createElement(a.a, {
                        activated: t,
                        mode: this.props.viewMode,
                        modifiers: ["has-outline", ...this.props.modifiers],
                        user: e
                    }))
                })))
            }
        }

        Object.defineProperty(o, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {modifiers: []}
        })
    }, OHSQ: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            }));
            var r = s("/G9H"), n = s("I8Ok"), i = s("tX/w"), a = s("vZz4"), o = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, l = {}.hasOwnProperty;
            r.createElement;
            var c = function (t) {
                function s() {
                    return this.renderHeaderStream = o(this.renderHeaderStream, this), this.render = o(this.render, this), s.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) l.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.render = function () {
                    var e;
                    return Object(n.div)({className: Object(i.a)("update-streams-v2", [null != this.props.currentStreamId ? "with-active" : void 0])}, Object(n.div)({className: "update-streams-v2__container"}, function () {
                        var t, s, r, n;
                        for (n = [], t = 0, s = (r = this.props.updateStreams).length; t < s; t++) e = r[t], n.push(this.renderHeaderStream({stream: e}));
                        return n
                    }.call(this)))
                }, s.prototype.renderHeaderStream = function (t) {
                    var s, r, o;
                    return r = t.stream, o = e.kebabCase(r.display_name), s = Object(i.a)("update-streams-v2__item", [o, r.is_featured ? "featured" : void 0, this.props.currentStreamId === r.id ? "active" : void 0]), s += " t-changelog-stream--" + o, Object(n.a)({
                        href: Object(a.c)(r.latest_build),
                        key: r.id,
                        className: s
                    }, Object(n.div)({className: "update-streams-v2__bar u-changelog-stream--bg"}), Object(n.p)({className: "update-streams-v2__row update-streams-v2__row--name"}, r.display_name), Object(n.p)({className: "update-streams-v2__row update-streams-v2__row--version"}, r.latest_build.display_version), r.user_count > 0 ? Object(n.p)({className: "update-streams-v2__row update-streams-v2__row--users"}, osu.transChoice("changelog.builds.users_online", r.user_count)) : void 0)
                }, s
            }(r.PureComponent)
        }).call(this, s("Hs9Z"))
    }, "OjW+": function (e, t, s) {
        "use strict";
        var r = s("oTtm"), n = s("MtWa"), i = s("0h6b"), a = s("KUml"), o = s("f4vq"), l = s("/G9H"), c = s("tX/w"),
            u = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };
        const d = {entered: {opacity: 1}, entering: {}, exited: {}, exiting: {}, unmounted: {}},
            p = Object(a.b)(({beatmaps: e}) => l.createElement("div", {className: "beatmaps-popup__group"}, e.map(e => l.createElement(m, {
                key: e.id,
                beatmap: e
            })))), m = Object(a.b)(({beatmap: e}) => l.createElement("a", {
                className: "beatmaps-popup-item",
                href: Object(i.a)("beatmaps.show", {beatmap: e.id})
            }, l.createElement("span", {className: "beatmaps-popup-item__col beatmaps-popup-item__col--mode"}, l.createElement("span", {className: `fal fa-extra-mode-${e.mode}`})), l.createElement(r.a, {rating: e.difficulty_rating}), l.createElement("span", {className: "beatmaps-popup-item__col beatmaps-popup-item__col--name u-ellipsis-overflow"}, e.version)));
        let h = class extends l.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "contentRef", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: l.createRef()
                })
            }

            render() {
                const e = Object.assign({
                    opacity: 0,
                    transitionDuration: `${this.props.transitionDuration}ms`
                }, d[this.props.state]);
                if (null != this.props.parent) {
                    const t = this.props.parent.getBoundingClientRect();
                    e.top = `${window.scrollY + t.bottom}px`, e.left = `${window.scrollX + t.left}px`, e.width = `${t.width}px`
                }
                return l.createElement(n.a, null, l.createElement("div", {
                    ref: this.contentRef,
                    className: Object(c.a)("beatmaps-popup", [`size-${o.a.userPreferences.get("beatmapset_card_size")}`]),
                    onMouseEnter: this.props.onMouseEnter,
                    onMouseLeave: this.props.onMouseLeave,
                    style: e
                }, l.createElement("div", {className: "beatmaps-popup__content"}, [...this.props.groupedBeatmaps].map(([e, t]) => t.length > 0 && l.createElement(p, {
                    key: e,
                    beatmaps: t
                })))))
            }
        };
        h = u([a.b], h), t.a = h
    }, P9aV: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("lv9K"), n = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class i {
            constructor(e) {
                Object.defineProperty(this, "core", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object(r.p)(this)
            }

            get blocks() {
                return null == this.core.currentUser ? new Set : new Set(this.core.currentUser.blocks.map(e => e.target_id))
            }

            get friends() {
                return null == this.core.currentUser ? new Map : new Map(this.core.currentUser.friends.map(e => [e.target_id, e]))
            }

            isFriendWith(e) {
                return null != this.friends.get(e)
            }
        }

        n([r.h], i.prototype, "blocks", null), n([r.h], i.prototype, "friends", null)
    }, "PVx+": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "d", (function () {
                return i
            })), s.d(t, "a", (function () {
                return a
            })), s.d(t, "b", (function () {
                return o
            })), s.d(t, "c", (function () {
                return c
            }));
            var r = s("0h6b"), n = s("lv9K");

            function i(e) {
                return Array.isArray(e) ? e.length : 0
            }

            const a = Object(n.f)((t, s, a) => {
                t.pagination.loading = !0;
                let o = a.limit;
                "number" == typeof o && Number.isFinite(o) || (o = 50);
                const c = o + 1, u = Object.assign(Object.assign({}, a), {limit: c, offset: i(t.items)});
                return e.ajax(Object(r.a)(s, u)).done(e => {
                    l(t, e, c)
                }).always(Object(n.f)(() => {
                    t.pagination.loading = !1
                }))
            }), o = (e, t) => a(e, "users.kudosu", {user: t}), l = Object(n.f)((e, t, s) => {
                Array.isArray(t) ? (Array.isArray(e.items) || (e.items = []), e.pagination.hasMore = c(s - 1, t), e.items.push(...t)) : 0 === i(e.items) && (e.items = t)
            }), c = Object(n.f)((e, t) => {
                if (!Array.isArray(t)) return !1;
                const s = t.length > e;
                return s && t.pop(), s
            })
        }).call(this, s("5wds"))
    }, PdfH: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return d
            }));
            var r = s("0h6b"), n = s("f4vq"), i = s("/G9H"), a = s("/jJF"), o = s("tX/w"), l = s("cX0L"), c = s("/HbY");
            const u = "user-action-button";

            class d extends i.Component {
                constructor(t) {
                    var s, o, c;
                    super(t), Object.defineProperty(this, "buttonRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: i.createRef()
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `follow-user-mapping-button-${Object(l.a)()}`
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "onClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setState({loading: !0}, () => {
                                const t = {
                                    data: {
                                        follow: {
                                            notifiable_id: this.props.userId,
                                            notifiable_type: "user",
                                            subtype: "mapping"
                                        }
                                    }
                                };
                                this.state.following ? (t.type = "DELETE", t.url = Object(r.a)("follows.destroy")) : (t.type = "POST", t.url = Object(r.a)("follows.store")), this.xhr = e.ajax(t).done(this.updateData).fail(Object(a.d)(this.buttonRef.current)).always(() => this.setState({loading: !1}))
                            })
                        }
                    }), Object.defineProperty(this, "refresh", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e, t;
                            this.setState({following: null !== (t = null === (e = n.a.currentUser) || void 0 === e ? void 0 : e.follow_user_mapping.includes(this.props.userId)) && void 0 !== t && t})
                        }
                    }), Object.defineProperty(this, "updateData", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            e.publish("user:followUserMapping:update", {
                                following: !this.state.following,
                                userId: this.props.userId
                            })
                        }
                    });
                    const u = null !== (o = null === (s = n.a.currentUser) || void 0 === s ? void 0 : s.follow_user_mapping.includes(this.props.userId)) && void 0 !== o && o;
                    let d = null !== (c = this.props.followers) && void 0 !== c ? c : 0;
                    !1 !== u && (d -= 1), this.state = {followersWithoutSelf: d, following: u, loading: !1}
                }

                componentDidMount() {
                    e.subscribe(`user:followUserMapping:refresh.${this.eventId}`, this.refresh)
                }

                componentWillUnmount() {
                    var t;
                    e.unsubscribe(`.${this.eventId}`), null === (t = this.xhr) || void 0 === t || t.abort()
                }

                render() {
                    const e = !(null == n.a.currentUser || n.a.currentUser.id === this.props.userId);
                    if (!e && !this.props.alwaysVisible) return null;
                    const t = e ? osu.trans(`follows.mapping.${this.state.following ? "to_0" : "to_1"}`) : osu.trans("follows.mapping.followers"),
                        s = Object(o.a)(u, this.props.modifiers, {friend: this.state.following}),
                        r = this.state.loading || !e;
                    return i.createElement("div", {title: t}, i.createElement("button", {
                        ref: this.buttonRef,
                        className: s,
                        disabled: r,
                        onClick: this.onClick
                    }, this.renderIcon(), this.renderCounter()))
                }

                followers() {
                    return this.state.followersWithoutSelf + (this.state.following ? 1 : 0)
                }

                renderCounter() {
                    if (null != this.props.showFollowerCounter && null != this.props.followers) return i.createElement("span", {className: `${u}__counter`}, osu.formatNumber(this.followers()))
                }

                renderIcon() {
                    const e = this.state.loading ? i.createElement(c.a, null) : i.createElement("i", {className: "fas fa-bell"});
                    return i.createElement("span", {className: `${u}__icon-container`}, e)
                }
            }
        }).call(this, s("5wds"))
    }, QUfv: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
            constructor(e) {
                Object.defineProperty(this, "message", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, R9Sp: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            }));
            var r = s("/G9H"), n = s("tX/w"), i = s("/HbY");
            const a = "show-more-link";

            class o extends r.PureComponent {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "onClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null == this.props.callback ? null == this.props.url && null != this.props.event && e.publish(this.props.event, this.props.data) : this.props.callback(this.props.data)
                        }
                    })
                }

                render() {
                    if (!this.props.hasMore && !this.props.loading) return null;
                    const e = {children: this.children(), className: Object(n.a)(a, this.props.modifiers)};
                    return this.props.loading ? r.createElement("span", Object.assign({"data-disabled": "1"}, e)) : null == this.props.url ? r.createElement("button", Object.assign({
                        onClick: this.onClick,
                        type: "button"
                    }, e)) : r.createElement("a", Object.assign({href: this.props.url, onClick: this.onClick}, e))
                }

                children() {
                    var e, t;
                    const s = r.createElement("span", {className: `fas fa-angle-${null !== (e = this.props.direction) && void 0 !== e ? e : "down"}`});
                    return r.createElement(r.Fragment, null, r.createElement("span", {className: `${a}__spinner`}, r.createElement(i.a, null)), r.createElement("span", {className: `${a}__label`}, r.createElement("span", {className: `${a}__label-icon ${a}__label-icon--left`}, s), r.createElement("span", {className: `${a}__label-text`}, null !== (t = this.props.label) && void 0 !== t ? t : osu.trans("common.buttons.show_more"), null != this.props.remaining && ` (${this.props.remaining})`), r.createElement("span", {className: `${a}__label-icon ${a}__label-icon--right`}, s)))
                }
            }

            Object.defineProperty(o, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {hasMore: !1, loading: !1}
            })
        }).call(this, s("5wds"))
    }, Rfpg: function (e, t, s) {
        "use strict";
        s.d(t, "b", (function () {
            return r
        })), s.d(t, "c", (function () {
            return n
        })), s.d(t, "a", (function () {
            return i
        }));
        const r = ["ANNOUNCE", "PUBLIC", "GROUP", "PM"], n = new Set(r);

        function i(e) {
            return e.filter(e => n.has(e.type))
        }
    }, SwPc: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        function i(e) {
            let t = Object(n.a)("mod", e.modifiers);
            return t += ` mod--${e.mod}`, r.createElement("div", {
                className: t,
                title: osu.trans(`beatmaps.mods.${e.mod}`)
            })
        }
    }, TezV: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("AqrC"), n = s("/G9H"), i = s("tX/w"), a = s("ss8h");

        class o extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "renderButton", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: (e, t, s) => {
                        var r;
                        const a = null !== (r = this.props.menuOptions.find(e => e.id === this.props.selected)) && void 0 !== r ? r : this.props.menuOptions[0],
                            o = [];
                        return this.props.disabled && (s = () => {
                        }, o.push("disabled")), n.createElement("div", {
                            ref: t,
                            className: Object(i.a)("icon-dropdown-menu", o),
                            contentEditable: !1,
                            onClick: s
                        }, a.icon, e)
                    }
                }), Object.defineProperty(this, "renderMenuItem", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => n.createElement("button", {
                        key: e.id,
                        className: Object(i.a)("simple-menu__item", {active: e.id === this.props.selected}),
                        "data-id": e.id,
                        onClick: this.select
                    }, n.createElement("div", {className: Object(i.a)("simple-menu__item-icon", "icon-dropdown-menu")}, e.icon), n.createElement("div", {className: "simple-menu__label"}, e.label))
                }), Object.defineProperty(this, "select", {
                    enumerable: !0, configurable: !0, writable: !0, value: e => {
                        var t;
                        e.preventDefault();
                        const s = e.currentTarget;
                        s && this.props.onSelect(null !== (t = s.dataset.id) && void 0 !== t ? t : "")
                    }
                })
            }

            render() {
                return n.createElement(r.a, {
                    customRender: this.renderButton,
                    direction: "right"
                }, () => n.createElement("div", {className: "simple-menu simple-menu--popup-menu-compact"}, this.props.menuOptions.map(this.renderMenuItem)))
            }
        }

        Object.defineProperty(o, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: a.a})
    }, TyNR: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return i
            }));
            var r = s("LPNJ"), n = s("/DQ7");

            class i {
                constructor() {
                    Object.defineProperty(this, "current", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: null
                    }), Object.defineProperty(this, "documentMouseEventTarget", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "close", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.show()
                        }
                    }), Object.defineProperty(this, "restoreSaved", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.show(document.body.dataset.clickMenuCurrent)
                        }
                    }), Object.defineProperty(this, "saveCurrent", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null == this.current ? delete document.body.dataset.clickMenuCurrent : document.body.dataset.clickMenuCurrent = this.current
                        }
                    }), Object.defineProperty(this, "show", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            var s, r;
                            const i = this.tree();
                            this.current = t;
                            const a = this.tree(), o = document.querySelectorAll(".js-click-menu[data-click-menu-id]");
                            let l = null, c = !1;
                            for (const e of o) {
                                if (!(e instanceof HTMLElement)) continue;
                                const t = e.dataset.clickMenuId;
                                null == t || -1 === a.indexOf(t) ? (Object(n.b)(e), e.classList.remove("js-click-menu--active"), null === (s = this.menuLink(t)) || void 0 === s || s.classList.remove("js-click-menu--active")) : (Object(n.a)(e), e.classList.add("js-click-menu--active"), null === (r = this.menuLink(t)) || void 0 === r || r.classList.add("js-click-menu--active"), c = !0, t === this.current && (l = e))
                            }
                            c || (this.current = null), e.publish("click-menu:current", {
                                previousTree: i,
                                target: this.current,
                                tree: a
                            });
                            const u = null == l ? void 0 : l.querySelector(".js-click-menu--autofocus");
                            u instanceof HTMLElement && u.focus()
                        }
                    }), Object.defineProperty(this, "toggle", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.currentTarget, s = this.tree();
                            e.preventDefault(), e.stopPropagation();
                            const r = t.dataset.clickMenuTarget;
                            let n = r;
                            if (null != r) {
                                const e = s.indexOf(r);
                                -1 !== e && (n = s[e + 1])
                            }
                            this.show(n)
                        }
                    }), Object.defineProperty(this, "tree", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (null == this.current) return [];
                            let e = this.current;
                            const t = [e];
                            for (; null != (e = this.closestMenuId(this.menuLink(e)));) t.push(e);
                            return t
                        }
                    }), Object.defineProperty(this, "onDocumentMousedown", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.documentMouseEventTarget = 0 === e.button ? e.target : null
                        }
                    }), Object.defineProperty(this, "onDocumentMouseup", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            this.documentMouseEventTarget === t.target && 0 === t.button && (Object(r.a)() || null != this.current && (e(t.target).closest(".js-click-menu").length > 0 || this.show()))
                        }
                    }), e(document).on("click", ".js-click-menu--close", this.close), e(document).on("click", ".js-click-menu[data-click-menu-target]", this.toggle), e(document).on("mousedown", this.onDocumentMousedown), e(document).on("mouseup", this.onDocumentMouseup), document.addEventListener("turbolinks:load", this.restoreSaved), document.addEventListener("turbolinks:before-cache", this.saveCurrent)
                }

                closestMenuId(t) {
                    if (null != t) return e(t).parents("[data-click-menu-id]").attr("data-click-menu-id")
                }

                menu(e) {
                    return document.querySelector(`.js-click-menu[data-click-menu-id${null == e ? "" : `='${e}'`}]`)
                }

                menuLink(e) {
                    return document.querySelector(`.js-click-menu[data-click-menu-target${null == e ? "" : `='${e}'`}]`)
                }
            }
        }).call(this, s("5wds"))
    }, UAat: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            }));
            var r = s("/G9H"), n = s("WKXC"), i = s("cX0L");
            const a = "notification-banner-v2";

            class o extends r.PureComponent {
                constructor(e) {
                    var t;
                    super(e), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `notification-banner-${Object(i.a)()}`
                    }), Object.defineProperty(this, "portalContainer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "removePortalContainer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.portalContainer.remove()
                        }
                    }), this.portalContainer = document.createElement("div");
                    const s = (null !== (t = window.newBody) && void 0 !== t ? t : document.body).querySelector(".js-notification-banners");
                    if (null == s) throw new Error("Notification banner container is missing");
                    s.appendChild(this.portalContainer)
                }

                componentDidMount() {
                    e(document).on(`turbolinks:before-cache.${this.eventId}`, this.removePortalContainer)
                }

                componentWillUnmount() {
                    e(document).off(`.${this.eventId}`), this.removePortalContainer()
                }

                render() {
                    return Object(n.createPortal)(this.renderNotification(), this.portalContainer)
                }

                renderNotification() {
                    return r.createElement("div", {className: `${a} ${a}--${this.props.type}`}, r.createElement("div", {className: `${a}__col ${a}__col--icon`}), r.createElement("div", {className: `${a}__col ${a}__col--label`}, r.createElement("div", {className: `${a}__type`}, this.props.type), r.createElement("div", {className: `${a}__text`}, this.props.title)), r.createElement("div", {className: `${a}__col`}, r.createElement("div", {className: `${a}__text`}, this.props.message)))
                }
            }
        }).call(this, s("5wds"))
    }, UBw1: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("7EfK"), n = s("/G9H"), i = function (e, t) {
            var s = {};
            for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (s[r] = e[r]);
            if (null != e && "function" == typeof Object.getOwnPropertySymbols) {
                var n = 0;
                for (r = Object.getOwnPropertySymbols(e); n < r.length; n++) t.indexOf(r[n]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[n]) && (s[r[n]] = e[r[n]])
            }
            return s
        };

        function a(e) {
            const {dateTime: t, format: s = "ll", relative: a = !1} = e, o = i(e, ["dateTime", "format", "relative"]),
                l = "string" == typeof t ? t : t.format();
            let c, u = l;
            if (a) c = "js-timeago"; else {
                c = "js-tooltip-time", u = ("string" == typeof t ? r(t) : t).format(s)
            }
            return n.createElement("time", Object.assign({className: c, dateTime: l, title: l}, o), u)
        }
    }, UZmH: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("lv9K"), n = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class i {
            constructor() {
                Object.defineProperty(this, "beatmapsetIds", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Set
                }), Object.defineProperty(this, "cursorString", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "fetchedAt", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "hasMore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "total", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), Object(r.p)(this)
            }

            get hasMoreForPager() {
                return this.hasMore && this.beatmapsetIds.size < this.total
            }

            get isExpired() {
                return null == this.fetchedAt || (new Date).getTime() - this.fetchedAt.getTime() > i.CACHE_DURATION_MS
            }

            append(e) {
                for (const t of e.beatmapsets) this.beatmapsetIds.add(t.id);
                this.cursorString = e.cursor_string, this.fetchedAt = new Date, this.hasMore = null !== e.cursor_string, this.total = e.total
            }

            reset() {
                this.beatmapsetIds.clear(), this.fetchedAt = void 0, this.cursorString = void 0, this.hasMore = !1, this.total = 0
            }
        }

        Object.defineProperty(i, "CACHE_DURATION_MS", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: 6e4
        }), n([r.q], i.prototype, "beatmapsetIds", void 0), n([r.q], i.prototype, "hasMore", void 0), n([r.q], i.prototype, "total", void 0), n([r.h], i.prototype, "hasMoreForPager", null), n([r.h], i.prototype, "isExpired", null), n([r.f], i.prototype, "append", null), n([r.f], i.prototype, "reset", null)
    }, UxfW: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            }));
            var r = s("/G9H"), n = s("xZyo"), i = s("yun6"), a = s("tX/w"), o = s("cX0L"), l = s("oTtm");

            class c extends r.Component {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "tooltipId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: ""
                    }), Object.defineProperty(this, "handleMouseOver", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            if (!this.props.withTooltip) return;
                            const s = t.currentTarget;
                            if (s._tooltip === this.tooltipId) return;
                            const i = e(Object(n.renderToStaticMarkup)(r.createElement("div", {className: "tooltip-beatmap"}, r.createElement("div", {className: "tooltip-beatmap__text"}, this.props.beatmap.version), r.createElement("div", {className: "tooltip-beatmap__badge"}, r.createElement(l.a, {rating: this.props.beatmap.difficulty_rating})))));
                            if (null != s._tooltip) return s._tooltip = this.tooltipId, void e(s).qtip("set", {"content.text": i});
                            s._tooltip = this.tooltipId;
                            const a = {
                                content: i,
                                hide: "touchstart" === t.type ? {
                                    event: "touchstart unfocus",
                                    inactive: 3e3
                                } : {event: "click mouseleave"},
                                overwrite: !1,
                                position: {at: "top center", my: "bottom center", viewport: e(window)},
                                show: {event: t.type, ready: !0},
                                style: {classes: "qtip qtip--tooltip-beatmap", tip: {height: 9, width: 10}}
                            };
                            e(s).qtip(a, t)
                        }
                    })
                }

                render() {
                    this.tooltipId = `beatmap-icon-${this.props.beatmap.id}-${Object(o.a)()}`;
                    const e = this.props.beatmap.convert && !this.props.showConvertMode ? "osu" : this.props.beatmap.mode,
                        t = Object(a.a)("beatmap-icon", this.props.modifiers, {"with-hover": this.props.withTooltip}),
                        s = {"--diff": Object(i.d)(this.props.beatmap.difficulty_rating)};
                    return r.createElement("div", {
                        className: t,
                        onMouseOver: this.handleMouseOver,
                        onTouchStart: this.handleMouseOver,
                        style: s
                    }, r.createElement("i", {className: `fal fa-extra-mode-${e}`}))
                }
            }

            Object.defineProperty(c, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {showConvertMode: !1, withTooltip: !1}
            })
        }).call(this, s("5wds"))
    }, V0Mc: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            }));
            var r = s("/G9H"), n = s("I8Ok"), i = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, a = {}.hasOwnProperty, o = function (t) {
                function s(e) {
                    this.reset = i(this.reset, this), this.render = i(this.render, this), this.mountObserver = i(this.mountObserver, this), this.onScroll = i(this.onScroll, this), this.onClick = i(this.onClick, this), this.componentWillUnmount = i(this.componentWillUnmount, this), s.__super__.constructor.call(this, e), this.state = {lastScrollY: null}
                }

                return function (e, t) {
                    for (var s in t) a.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.componentWillUnmount = function () {
                    if (document.removeEventListener("scroll", this.onScroll), null != this.observer) return this.observer.disconnect(), this.observer = null
                }, s.prototype.onClick = function (t) {
                    var s, r;
                    return null != this.state.lastScrollY ? (window.scrollTo(window.pageXOffset, this.state.lastScrollY), this.setState({lastScrollY: null})) : (r = null != (null != (s = this.props.anchor) ? s.current : void 0) ? e(this.props.anchor.current).offset().top : 0, window.pageYOffset > r ? (this.setState({lastScrollY: window.pageYOffset}), window.scrollTo(window.pageXOffset, r), this.mountObserver()) : void 0)
                }, s.prototype.onScroll = function (e) {
                    if (this.setState({lastScrollY: null}), document.removeEventListener("scroll", this.onScroll), null != this.observer) return this.observer.disconnect(), this.observer = null
                }, s.prototype.mountObserver = function () {
                    var e, t, s, r;
                    return null != window.IntersectionObserver ? (s = null != (e = null != (t = this.props.anchor) ? t.current : void 0) ? e : document.body, this.observer = new IntersectionObserver((r = this, function (e) {
                        var t, n, i, a;
                        for (a = [], n = 0, i = e.length; n < i; n++) {
                            if ((t = e[n]).target === s && 0 === t.boundingClientRect.top) {
                                document.addEventListener("scroll", r.onScroll);
                                break
                            }
                            a.push(void 0)
                        }
                        return a
                    })), this.observer.observe(s)) : Timeout.set(0, function (e) {
                        return function () {
                            return document.addEventListener("scroll", e.onScroll)
                        }
                    }(this))
                }, s.prototype.render = function () {
                    return Object(n.button)({
                        className: "back-to-top",
                        "data-tooltip-float": "fixed",
                        onClick: this.onClick,
                        title: null != this.state.lastScrollY ? osu.trans("common.buttons.back_to_previous") : osu.trans("common.buttons.back_to_top")
                    }, Object(n.i)({className: null != this.state.lastScrollY ? "fas fa-angle-down" : "fas fa-angle-up"}))
                }, s.prototype.reset = function () {
                    return this.setState({lastScrollY: null})
                }, s
            }(r.PureComponent)
        }).call(this, s("5wds"))
    }, VEn2: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return c
        }));
        var r = s("7EfK"), n = s("/G9H"), i = s("0h6b"), a = s("f4vq");

        class o extends n.Component {
            constructor(e) {
                if (super(e), Object.defineProperty(this, "state", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {expanded: !1}
                }), Object.defineProperty(this, "stateRecord", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "recordState", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        null != this.stateRecord && (this.stateRecord.dataset[this.stateRecordKey] = this.state.expanded ? "1" : "")
                    }
                }), Object.defineProperty(this, "renderPost", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        let t = "news-sidebar-month__item";
                        return null != this.props.currentPost && this.props.currentPost.id === e.id && (t += " news-sidebar-month__item--active"), n.createElement("li", {key: e.id}, n.createElement("a", {
                            className: t,
                            href: Object(i.a)("news.show", {news: e.slug})
                        }, e.title))
                    }
                }), Object.defineProperty(this, "toggleExpand", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({expanded: !this.state.expanded}, this.recordState)
                    }
                }), a.a.windowSize.isMobile) this.state.expanded = !1; else if (null == e.currentPost) this.state.expanded = e.initialExpand; else {
                    const t = r.utc(e.currentPost.published_at);
                    this.state.expanded = t.year() === e.date.year() && t.month() === e.date.month()
                }
                const t = document.querySelector(".js-news-sidebar-record");
                t instanceof HTMLElement && (this.stateRecord = t), null == this.stateRecordValue ? this.recordState() : this.state.expanded = this.stateRecordValue
            }

            get stateRecordKey() {
                return this.props.date.format("YYYYMM")
            }

            get stateRecordValue() {
                return this.stateRecord instanceof HTMLElement && null != this.stateRecord.dataset[this.stateRecordKey] ? "1" === this.stateRecord.dataset[this.stateRecordKey] : null
            }

            render() {
                return n.createElement("div", {className: "news-sidebar-month"}, n.createElement("button", {
                    className: "news-sidebar-month__toggle",
                    onClick: this.toggleExpand,
                    type: "button"
                }, this.props.date.format(osu.trans("common.datetime.year_month_short.moment")), n.createElement("span", {className: "news-sidebar-month__toggle-icon"}, n.createElement("i", {className: this.state.expanded ? "fas fa-chevron-up" : "fas fa-chevron-down"}))), n.createElement("ul", {className: `news-sidebar-month__items ${this.state.expanded ? "" : "hidden"}`}, this.props.posts.map(this.renderPost)))
            }
        }

        function l(e) {
            return n.createElement("div", {className: "news-sidebar-years"}, e.years.map(t => n.createElement("a", {
                key: t,
                className: `news-sidebar-years__item ${t === e.currentYear ? "news-sidebar-years__item--active" : ""}`,
                href: Object(i.a)("news.index", {year: t})
            }, t)))
        }

        function c(e) {
            var t;
            const s = {}, i = {}, a = new Set;
            for (const n of e.data.news_posts) {
                const e = r.utc(n.published_at), o = e.format("YYYY-MM");
                (s[o] = null !== (t = s[o]) && void 0 !== t ? t : []).push(n), i[o] = e, a.add(o)
            }
            const c = [...a];
            c.sort().reverse();
            let u = !0;
            return n.createElement("div", {className: "sidebar"}, n.createElement("button", {
                className: "sidebar__mobile-toggle sidebar__mobile-toggle--mobile-only js-mobile-toggle",
                "data-mobile-toggle-target": "news-archive",
                type: "button"
            }, n.createElement("h2", {className: "sidebar__title"}, osu.trans("news.sidebar.archive")), n.createElement("div", {className: "sidebar__mobile-toggle-icon"}, n.createElement("i", {className: "fas fa-chevron-down"}))), n.createElement("div", {
                className: "sidebar__content hidden-xs js-mobile-toggle",
                "data-mobile-toggle-id": "news-archive"
            }, n.createElement(l, {currentYear: e.data.current_year, years: e.data.years}), c.map(t => {
                if (null == s[t] || null == i[t]) return;
                const r = i[t], a = u;
                return u = !1, n.createElement(o, {
                    key: t,
                    currentPost: e.currentPost,
                    date: r,
                    initialExpand: a,
                    posts: s[t]
                })
            })))
        }
    }, VbpL: function (e, t, s) {
        "use strict";
        (function (e) {
            var r = s("kXXC"), n = s("WLnA"), i = s("0h6b"), a = s("lv9K"), o = s("KUml"), l = s("f4vq"), c = s("/G9H"),
                u = s("/jJF"), d = s("tX/w"), p = s("/HbY"), m = function (e, t, s, r) {
                    var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };
            const h = "user-action-button";
            let b = class extends c.Component {
                constructor(t) {
                    var s;
                    super(t), Object.defineProperty(this, "followersWithoutSelf", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "loading", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "clicked", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.loading = !0, null == this.friend ? this.xhr = e.ajax(Object(i.a)("friends.store", {target: this.props.userId}), {type: "POST"}) : this.xhr = e.ajax(Object(i.a)("friends.destroy", {friend: this.props.userId}), {type: "DELETE"}), this.xhr.done(this.updateFriends).fail(Object(u.c)(this.clicked)).always(Object(a.f)(() => this.loading = !1))
                        }
                    }), Object.defineProperty(this, "updateFriends", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            null != l.a.currentUser && (l.a.currentUser.friends = t, e.publish("user:update", l.a.currentUser), Object(n.a)(new r.a(this.props.userId)))
                        }
                    }), this.followersWithoutSelf = null !== (s = this.props.followers) && void 0 !== s ? s : 0, null != this.friend && (this.followersWithoutSelf -= 1), Object(a.p)(this)
                }

                get followers() {
                    return this.followersWithoutSelf + (null == this.friend ? 0 : 1)
                }

                get friend() {
                    return l.a.currentUserModel.friends.get(this.props.userId)
                }

                get isFriendLimit() {
                    return null == l.a.currentUser || l.a.currentUser.friends.length >= l.a.currentUser.max_friends
                }

                get isVisible() {
                    return null != l.a.currentUser && Number.isFinite(this.props.userId) && this.props.userId !== l.a.currentUser.id && !l.a.currentUser.blocks.some(e => e.target_id === this.props.userId)
                }

                get showFollowerCounter() {
                    return null != this.props.followers
                }

                get title() {
                    return this.isVisible ? null != this.friend ? osu.trans("friends.buttons.remove") : this.isFriendLimit ? osu.trans("friends.too_many") : osu.trans("friends.buttons.add") : osu.trans("friends.buttons.disabled")
                }

                componentWillUnmount() {
                    var e;
                    null === (e = this.xhr) || void 0 === e || e.abort()
                }

                render() {
                    if (!this.props.alwaysVisible && !this.isVisible) return null;
                    const e = null == this.friend || this.loading ? null : this.friend.mutual ? "mutual" : "friend",
                        t = Object(d.a)(h, this.props.modifiers, e),
                        s = !this.isVisible || this.loading || this.isFriendLimit && null == this.friend;
                    return c.createElement("div", {title: this.title}, c.createElement("button", {
                        className: t,
                        disabled: s,
                        onClick: this.clicked,
                        type: "button"
                    }, c.createElement("span", {className: `${h}__icon-container`}, this.renderIcon()), this.renderCounter()))
                }

                renderCounter() {
                    if (this.showFollowerCounter) return c.createElement("span", {className: `${h}__counter`}, osu.formatNumber(this.followers))
                }

                renderIcon() {
                    return this.loading ? c.createElement(p.a, null) : this.isVisible ? null != this.friend ? c.createElement(c.Fragment, null, c.createElement("span", {className: `${h}__icon ${h}__icon--hover-visible`}, c.createElement("span", {className: "fas fa-user-times"})), this.friend.mutual ? c.createElement("span", {className: `${h}__icon ${h}__icon--hover-hidden`}, c.createElement("span", {className: "fas fa-user-friends"})) : c.createElement("span", {className: `${h}__icon ${h}__icon--hover-hidden`}, c.createElement("span", {className: "fas fa-user"}))) : c.createElement("span", {className: this.isFriendLimit ? "fas fa-user" : "fas fa-user-plus"}) : c.createElement("span", {className: "fas fa-user"})
                }
            };
            Object.defineProperty(b, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {alwaysVisible: !1}
            }), m([a.q], b.prototype, "followersWithoutSelf", void 0), m([a.q], b.prototype, "loading", void 0), m([a.h], b.prototype, "followers", null), m([a.h], b.prototype, "friend", null), m([a.h], b.prototype, "isFriendLimit", null), m([a.h], b.prototype, "isVisible", null), m([a.h], b.prototype, "title", null), m([a.f], b.prototype, "clicked", void 0), m([a.f], b.prototype, "updateFriends", void 0), b = m([o.b], b), t.a = b
        }).call(this, s("5wds"))
    }, VsY1: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return ye
        }));
        var r = s("va1x");

        class n {
            constructor(e) {
                Object.defineProperty(this, "channelId", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }

        var i = s("kXXC"), a = s("QUfv"), o = s("G27q"), l = s("WLnA"), c = s("Rfpg"), u = s("Hs9Z"), d = s("lv9K"),
            p = s("205K"), m = s("rqcF"), h = s("bfq+");

        class b {
            constructor(e) {
                Object.defineProperty(this, "json", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }

        var f = s("8gxX");

        class v {
            constructor(e) {
                Object.defineProperty(this, "channelStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "lastHistoryId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "retryDelay", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new f.a(0, 45e3)
                }), Object.defineProperty(this, "timerId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "xhr", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "ping", {
                    enumerable: !0, configurable: !0, writable: !0, value: () => {
                        this.xhr = Object(h.a)(this.channelStore.lastReceivedMessageId, this.lastHistoryId).done(e => {
                            var t;
                            const s = null === (t = Object(u.maxBy)(e.silences, "id")) || void 0 === t ? void 0 : t.id;
                            null != s && (this.lastHistoryId = s), Object(l.a)(new b(e.silences)), this.retryDelay.reset(), this.scheduleNextPing()
                        }).fail(e => {
                            401 !== e.status && this.scheduleNextPing()
                        })
                    }
                })
            }

            start() {
                null == this.timerId && this.scheduleNextPing()
            }

            stop() {
                var e;
                null === (e = this.xhr) || void 0 === e || e.abort(), this.retryDelay.reset(), null != this.timerId && (window.clearTimeout(this.timerId), this.timerId = void 0)
            }

            scheduleNextPing() {
                this.timerId = window.setTimeout(this.ping, this.retryDelay.get())
            }
        }

        var g = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }, y = function (e, t, s, r) {
            return new (s || (s = Promise))((function (n, i) {
                function a(e) {
                    try {
                        l(r.next(e))
                    } catch (t) {
                        i(t)
                    }
                }

                function o(e) {
                    try {
                        l(r.throw(e))
                    } catch (t) {
                        i(t)
                    }
                }

                function l(e) {
                    var t;
                    e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                        e(t)
                    }))).then(a, o)
                }

                l((r = r.apply(e, t || [])).next())
            }))
        };
        let w = class {
            constructor(e) {
                Object.defineProperty(this, "channelStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "isChatMounted", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isReady", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "selectedBoxed", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.box(0)
                }), Object.defineProperty(this, "skipRefresh", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isConnected", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "lastHistoryId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "pingService", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "selectedIndex", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), this.pingService = new v(e), Object(d.p)(this), Object(d.r)(e.channels, e => {
                    "delete" === e.type && this.refocusSelectedChannel()
                }), Object(d.g)(() => {
                    this.isReady && this.isChatMounted ? (this.pingService.start(), Object(l.a)(new a.a({event: "chat.start"}))) : (this.pingService.stop(), Object(l.a)(new a.a({event: "chat.end"})))
                }), Object(d.g)(() => y(this, void 0, void 0, (function* () {
                    this.isConnected && this.isChatMounted && (this.skipRefresh ? this.skipRefresh = !1 : yield this.updateChannelList(), Object(d.u)(() => {
                        this.channelStore.loadChannel(this.selected), this.isReady = !0
                    }))
                })))
            }

            get selected() {
                return this.selectedBoxed.get()
            }

            set selected(e) {
                this.selectedBoxed.set(e)
            }

            get selectedChannel() {
                return this.channelStore.get(this.selected)
            }

            get channelList() {
                return c.b.flatMap(e => this.channelStore.groupedChannels[e])
            }

            handleDispatchAction(e) {
                e instanceof p.a ? this.handleChatChannelJoinEvent(e) : e instanceof m.a ? this.handleChatChannelPartEvent(e) : e instanceof n ? this.handleChatNewConversationAdded(e) : e instanceof i.a ? this.handleFriendUpdated(e) : e instanceof o.a && this.handleSocketStateChanged(e)
            }

            selectChannel(e) {
                if (this.selected === e) return;
                null != this.selectedChannel && this.channelStore.markAsRead(this.selectedChannel.channelId);
                const t = this.channelStore.get(e);
                null != t ? (this.selected = e, this.selectedIndex = this.channelList.indexOf(t), this.channelStore.loadChannel(e)) : console.error(`Trying to switch to non-existent channel ${e}`)
            }

            selectFirst() {
                0 !== this.channelList.length && this.selectChannel(this.channelList[0].channelId)
            }

            focusChannelAtIndex(e) {
                if (0 === this.channelList.length) return;
                const t = Object(u.clamp)(e, 0, this.channelList.length - 1), s = this.channelList[t];
                this.selectChannel(s.channelId)
            }

            handleChatChannelJoinEvent(e) {
                this.channelStore.getOrCreate(e.json.channel_id).updateWithJson(e.json)
            }

            handleChatChannelPartEvent(e) {
                this.channelStore.partChannel(e.channelId, !1)
            }

            handleChatNewConversationAdded(e) {
                this.selectChannel(e.channelId)
            }

            handleFriendUpdated(e) {
                if (!this.isChatMounted) return;
                const t = this.channelStore.groupedChannels.PM.find(t => t.pmTarget === e.userId);
                null == t || t.refresh()
            }

            handleSocketStateChanged(e) {
                this.isConnected = e.connected, e.connected || (this.channelStore.channels.forEach(e => e.needsRefresh = !0), this.isReady = !1)
            }

            refocusSelectedChannel() {
                null != this.selectedChannel ? this.selectChannel(this.selectedChannel.channelId) : this.focusChannelAtIndex(this.selectedIndex)
            }

            updateChannelList() {
                return y(this, void 0, void 0, (function* () {
                    const e = yield Object(h.d)(this.channelStore.lastReceivedMessageId, this.lastHistoryId);
                    e && Object(d.u)(() => {
                        var t;
                        const s = null === (t = Object(u.maxBy)(e.silences, "id")) || void 0 === t ? void 0 : t.id;
                        null != s && (this.lastHistoryId = s), this.channelStore.updateWithJson(e)
                    })
                }))
            }
        };
        g([d.q], w.prototype, "isChatMounted", void 0), g([d.q], w.prototype, "isReady", void 0), g([d.q], w.prototype, "selectedBoxed", void 0), g([d.q], w.prototype, "isConnected", void 0), g([d.h], w.prototype, "selected", null), g([d.h], w.prototype, "selectedChannel", null), g([d.h], w.prototype, "channelList", null), g([d.f], w.prototype, "selectChannel", null), g([d.f], w.prototype, "selectFirst", null), g([d.f], w.prototype, "focusChannelAtIndex", null), g([d.f], w.prototype, "handleChatChannelJoinEvent", null), g([d.f], w.prototype, "handleChatChannelPartEvent", null), g([d.f], w.prototype, "handleChatNewConversationAdded", null), g([d.f], w.prototype, "handleFriendUpdated", null), g([d.f], w.prototype, "handleSocketStateChanged", null), g([d.f], w.prototype, "refocusSelectedChannel", null), g([d.f], w.prototype, "updateChannelList", null);
        var _ = w = g([l.b], w), O = s("bpgk"), j = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class E {
            constructor() {
                Object.defineProperty(this, "beatmapsets", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.map()
                }), Object(d.p)(this)
            }

            get(e) {
                return this.beatmapsets.get(e)
            }

            handleDispatchAction(e) {
                e instanceof O.a && this.flushStore()
            }

            update(e) {
                this.beatmapsets.set(e.id, e)
            }

            flushStore() {
                this.beatmapsets.clear()
            }
        }

        j([d.q], E.prototype, "beatmapsets", void 0), j([d.f], E.prototype, "update", null), j([d.f], E.prototype, "flushStore", null);
        var P = s("XvDi"), k = s("2hxc"), N = s("sTr9"), S = s("d+cC"), T = s("f4vq"), C = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }, x = function (e, t, s, r) {
            return new (s || (s = Promise))((function (n, i) {
                function a(e) {
                    try {
                        l(r.next(e))
                    } catch (t) {
                        i(t)
                    }
                }

                function o(e) {
                    try {
                        l(r.throw(e))
                    } catch (t) {
                        i(t)
                    }
                }

                function l(e) {
                    var t;
                    e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                        e(t)
                    }))).then(a, o)
                }

                l((r = r.apply(e, t || [])).next())
            }))
        };

        function M(e, t) {
            return e.name.localeCompare(t.name)
        }

        const R = {
            ANNOUNCE: M,
            GROUP: M,
            PM: (e, t) => e.newPmChannel ? -1 : t.newPmChannel ? 1 : e.lastMessageId === t.lastMessageId ? 0 : e.lastMessageId > t.lastMessageId ? -1 : 1,
            PUBLIC: M
        };
        let D = class {
            constructor() {
                Object.defineProperty(this, "channels", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.map()
                }), Object.defineProperty(this, "lastReceivedMessageId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), Object.defineProperty(this, "markingAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {}
                }), Object(d.p)(this)
            }

            get groupedChannels() {
                const e = function () {
                    const e = {};
                    for (const t of c.b) e[t] = [];
                    return e
                }();
                for (const t of this.channels.values()) null != t.supportedType && e[t.supportedType].push(t);
                for (const t of c.b) e[t] = e[t].sort(R[t]);
                return e
            }

            addNewConversation(e, t) {
                const s = this.getOrCreate(e.channel_id);
                return s.updateWithJson(e), s.addMessages([S.a.fromJson(t)]), s
            }

            findPM(e) {
                var t;
                if (e === (null === (t = T.a.currentUser) || void 0 === t ? void 0 : t.id)) return null;
                for (const [, s] of this.channels) if ("PM" === s.type && s.users.includes(e)) return s;
                return null
            }

            flushStore() {
                this.channels.clear()
            }

            get(e) {
                return this.channels.get(e)
            }

            getOrCreate(e) {
                let t = this.channels.get(e);
                return t || (t = new N.a(e), this.channels.set(e, t)), t
            }

            handleDispatchAction(e) {
                e instanceof k.a ? this.handleChatMessageNewEvent(e) : e instanceof P.a ? this.handleChatMessageSendAction(e) : e instanceof b && this.handleChatUpdateSilences(e)
            }

            loadChannel(e) {
                var t;
                null === (t = this.channels.get(e)) || void 0 === t || t.load()
            }

            loadChannelEarlierMessages(e) {
                var t;
                null === (t = this.get(e)) || void 0 === t || t.loadEarlierMessages()
            }

            markAsRead(e) {
                const t = this.get(e);
                if (null == t || !t.isUnread || !t.uiState.autoScroll) return;
                if (null != this.markingAsRead[e]) return;
                t.markAsRead();
                const s = window.setTimeout(Object(d.f)(() => {
                    var r, n;
                    this.markingAsRead[e] === s && delete this.markingAsRead[e], (null === (r = t.lastMessage) || void 0 === r ? void 0 : r.sender.id) !== (null === (n = T.a.currentUser) || void 0 === n ? void 0 : n.id) && Object(h.e)(t.channelId, t.lastMessageId)
                }), 1e3);
                this.markingAsRead[e] = s
            }

            partChannel(e, t = !0) {
                e > 0 && t && Object(h.g)(e, T.a.currentUserOrFail.id), this.channels.delete(e)
            }

            updateWithJson(e) {
                var t;
                if (null != e.presence && this.updateWithPresence(e.presence), null != e.messages) {
                    this.updateLastReceivedMessageId(e.messages);
                    const s = Object(u.groupBy)(e.messages, "channel_id");
                    for (const e of Object.keys(s)) {
                        const r = parseInt(e, 10), n = s[r].map(S.a.fromJson);
                        null === (t = this.channels.get(r)) || void 0 === t || t.addMessages(n)
                    }
                }
                null != e.silences && Object(l.a)(new b(e.silences))
            }

            updateWithPresence(e) {
                Object(c.a)(e).forEach(e => {
                    this.getOrCreate(e.channel_id).updateWithJson(e)
                }), this.channels.forEach(t => {
                    t.newPmChannel || e.find(e => e.channel_id === t.channelId) || this.channels.delete(t.channelId)
                })
            }

            handleChatMessageNewEvent(e) {
                for (const t of e.json.messages) {
                    const e = this.channels.get(t.channel_id);
                    null != e && e.addMessage(t)
                }
                this.updateLastReceivedMessageId(e.json.messages)
            }

            handleChatMessageSendAction(e) {
                return x(this, void 0, void 0, (function* () {
                    const t = e.message, s = this.getOrCreate(t.channelId);
                    s.addSendingMessage(t);
                    try {
                        if (s.newPmChannel) {
                            const e = s.users.slice().find(e => e !== T.a.currentUserOrFail.id);
                            if (null == e) return void console.debug("sendMessage:: userId not found?? this shouldn't happen");
                            const r = yield Object(h.f)(e, t);
                            Object(d.u)(() => {
                                this.channels.delete(t.channelId);
                                const e = this.addNewConversation(r.channel, r.message);
                                Object(l.a)(new n(e.channelId))
                            })
                        } else {
                            const e = yield Object(h.h)(t);
                            s.afterSendMesssage(t, e)
                        }
                    } catch (r) {
                        s.afterSendMesssage(t, null), osu.ajaxError(r)
                    }
                }))
            }

            handleChatUpdateSilences(e) {
                const t = new Set(e.json.map(e => e.user_id));
                this.removePublicMessagesFromUserIds(t)
            }

            removePublicMessagesFromUserIds(e) {
                this.groupedChannels.PUBLIC.forEach(t => {
                    t.removeMessagesFromUserIds(e)
                })
            }

            updateLastReceivedMessageId(e) {
                var t, s;
                null != e && (this.lastReceivedMessageId = null !== (s = null === (t = Object(u.maxBy)(e, "message_id")) || void 0 === t ? void 0 : t.message_id) && void 0 !== s ? s : this.lastReceivedMessageId)
            }
        };
        C([d.q], D.prototype, "channels", void 0), C([d.h], D.prototype, "groupedChannels", null), C([d.f], D.prototype, "addNewConversation", null), C([d.f], D.prototype, "flushStore", null), C([d.f], D.prototype, "getOrCreate", null), C([d.f], D.prototype, "loadChannel", null), C([d.f], D.prototype, "loadChannelEarlierMessages", null), C([d.f], D.prototype, "markAsRead", null), C([d.f], D.prototype, "partChannel", null), C([d.f], D.prototype, "updateWithJson", null), C([d.f], D.prototype, "updateWithPresence", null), C([d.f], D.prototype, "handleChatMessageNewEvent", null), C([d.f], D.prototype, "handleChatMessageSendAction", null), C([d.f], D.prototype, "handleChatUpdateSilences", null), C([d.f], D.prototype, "removePublicMessagesFromUserIds", null), C([d.f], D.prototype, "updateLastReceivedMessageId", null);
        var I = D = C([l.b], D), H = s("Cfan"), A = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class U {
            constructor() {
                Object.defineProperty(this, "clients", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object(d.p)(this)
            }

            handleDispatchAction(e) {
                e instanceof O.a && this.flushStore()
            }

            initialize(e) {
                for (const t of e) {
                    const e = new H.a(t);
                    this.clients.set(e.id, e)
                }
            }

            flushStore() {
                this.clients.clear()
            }
        }

        A([d.q], U.prototype, "clients", void 0), A([d.f], U.prototype, "flushStore", null);
        var B = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class L {
            constructor(e) {
                Object.defineProperty(this, "commentableId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "commentableType", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "createdAt", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "deletedAt", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "deletedById", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "editedAt", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "editedById", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "id", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "legacyName", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "message", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "messageHtml", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "parentId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "pinned", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "repliesCount", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "updatedAt", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "userId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "votesCount", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), this.commentableId = e.commentable_id, this.commentableType = e.commentable_type, this.createdAt = e.created_at, this.deletedAt = e.deleted_at, this.deletedById = e.deleted_by_id, this.editedAt = e.edited_at, this.editedById = e.edited_by_id, this.id = e.id, this.legacyName = e.legacy_name, this.message = e.message, this.messageHtml = e.message_html, this.parentId = e.parent_id, this.pinned = e.pinned, this.repliesCount = e.replies_count, this.updatedAt = e.updated_at, this.userId = e.user_id, this.votesCount = e.votes_count, Object(d.p)(this)
            }

            get canDelete() {
                return this.canModerate || this.isOwner
            }

            get canEdit() {
                return this.canModerate || this.isOwner && !this.isDeleted
            }

            get canHaveVote() {
                return !this.isDeleted
            }

            get canModerate() {
                return null != T.a.currentUser && (T.a.currentUser.is_admin || T.a.currentUser.is_moderator)
            }

            get canPin() {
                if (null == T.a.currentUser || null != this.parentId && !this.pinned) return !1;
                if (T.a.currentUser.is_admin) return !0;
                if ("beatmapset" !== this.commentableType || !this.pinned && T.a.dataStore.uiState.comments.pinnedCommentIds.length > 0) return !1;
                if (this.canModerate) return !0;
                if (!this.isOwner) return !1;
                const e = T.a.dataStore.commentableMetaStore.get(this.commentableType, this.commentableId);
                return null != e && "owner_id" in e && e.owner_id === T.a.currentUser.id
            }

            get canReport() {
                return null != T.a.currentUser && this.userId !== T.a.currentUser.id
            }

            get canRestore() {
                return this.canModerate
            }

            get canVote() {
                return !this.isOwner
            }

            get isDeleted() {
                return null != this.deletedAt
            }

            get isEdited() {
                return null != this.editedAt
            }

            get isOwner() {
                return null != T.a.currentUser && this.userId === T.a.currentUser.id
            }
        }

        B([d.h], L.prototype, "canDelete", null), B([d.h], L.prototype, "canEdit", null), B([d.h], L.prototype, "canHaveVote", null), B([d.h], L.prototype, "canModerate", null), B([d.h], L.prototype, "canPin", null), B([d.h], L.prototype, "canReport", null), B([d.h], L.prototype, "canRestore", null), B([d.h], L.prototype, "canVote", null), B([d.h], L.prototype, "isDeleted", null), B([d.h], L.prototype, "isEdited", null), B([d.h], L.prototype, "isOwner", null);
        var $ = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class q {
            constructor() {
                Object.defineProperty(this, "comments", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.map()
                }), Object.defineProperty(this, "userVotes", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Set
                }), Object.defineProperty(this, "groupedByParentId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {}
                }), Object(d.p)(this)
            }

            addUserVote(e) {
                this.userVotes.add(e.id)
            }

            addVoted(e) {
                null != e && e.forEach(e => this.userVotes.add(e))
            }

            flushStore() {
                this.invalidate(), this.comments.clear(), this.userVotes.clear()
            }

            getRepliesByParentId(e) {
                return this.groupedByParentId[String(e)]
            }

            initialize(e, t) {
                this.flushStore(), this.addVoted(t), this.updateWithJson(e)
            }

            removeUserVote(e) {
                this.userVotes.delete(e.id)
            }

            updateWithJson(e) {
                if (null != e) for (const t of e) {
                    const e = new L(t), s = this.comments.has(e.id);
                    if (this.comments.set(e.id, e), !s) {
                        const t = String(e.parentId);
                        null != this.groupedByParentId[t] ? this.groupedByParentId[t].push(e) : this.groupedByParentId[t] = [e]
                    }
                }
            }

            invalidate() {
                this.groupedByParentId = {}
            }
        }

        $([d.q], q.prototype, "comments", void 0), $([d.q], q.prototype, "userVotes", void 0), $([d.f], q.prototype, "addUserVote", null), $([d.f], q.prototype, "addVoted", null), $([d.f], q.prototype, "flushStore", null), $([d.f], q.prototype, "initialize", null), $([d.f], q.prototype, "removeUserVote", null), $([d.f], q.prototype, "updateWithJson", null);
        var G = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class F {
            constructor() {
                Object.defineProperty(this, "meta", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.map()
                }), Object(d.p)(this)
            }

            flushStore() {
                this.meta.clear()
            }

            get(e, t) {
                const s = this.meta.get(`${e}-${t}`);
                return null != s ? s : this.meta.get(null)
            }

            initialize(e) {
                this.flushStore(), this.updateWithJson(e)
            }

            updateWithJson(e) {
                if (null != e) for (const t of e) {
                    const e = "id" in t ? `${t.type}-${t.id}` : null;
                    this.meta.set(e, t)
                }
            }
        }

        G([d.q], F.prototype, "meta", void 0), G([d.f], F.prototype, "flushStore", null), G([d.f], F.prototype, "initialize", null), G([d.f], F.prototype, "updateWithJson", null);
        var V = s("uW+8"), W = s("nxXY"), K = s("3Zv4"), z = s("y2EG"), J = s("5evE"), X = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class Z {
            constructor(e, t, s, r) {
                Object.defineProperty(this, "objectId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "objectType", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: t
                }), Object.defineProperty(this, "category", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: s
                }), Object.defineProperty(this, "resolver", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: r
                }), Object.defineProperty(this, "cursor", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "displayOrder", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), Object.defineProperty(this, "isDeleting", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isLoading", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isMarkingAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "notifications", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object.defineProperty(this, "total", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: 0
                }), Object.defineProperty(this, "lastNotification", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object(d.p)(this)
            }

            get canMarkAsRead() {
                for (const [, e] of this.notifications) if (e.canMarkRead) return !0;
                return !1
            }

            get first() {
                var e;
                return null !== (e = this.orderedNotifications[0]) && void 0 !== e ? e : this.lastNotification
            }

            get hasMore() {
                return !(this.notifications.size >= this.total || null == this.cursor)
            }

            get hasVisibleNotifications() {
                return this.notifications.size > 0
            }

            get id() {
                return `${this.objectType}-${this.objectId}-${this.category}`
            }

            get identity() {
                return {category: this.category, objectId: this.objectId, objectType: this.objectType}
            }

            get isSingle() {
                return 1 === this.total
            }

            get orderedNotifications() {
                return [...this.notifications.values()].sort((e, t) => t.id - e.id)
            }

            get type() {
                return this.objectType
            }

            static fromJson(e, t) {
                const s = new Z(e.object_id, e.object_type, Object(J.a)(e.name), t);
                return s.updateWithJson(e), s
            }

            add(e) {
                this.notifications.set(e.id, e), this.displayOrder = Math.max(e.id, this.displayOrder)
            }

            delete() {
                this.resolver.delete(this)
            }

            deleteItem(e) {
                null != e && this.notifications.has(e.id) && this.resolver.delete(e)
            }

            loadMore(e) {
                null != this.cursor && (this.isLoading = !0, this.resolver.loadMore(this.identity, e, this.cursor).always(Object(d.f)(() => {
                    this.isLoading = !1
                })))
            }

            markAsRead(e) {
                null != e && this.notifications.has(e.id) && this.resolver.queueMarkAsRead(e)
            }

            markStackAsRead() {
                this.resolver.queueMarkAsRead(this)
            }

            remove(e) {
                1 === this.notifications.size && (this.lastNotification = this.notifications.values().next().value);
                const t = this.notifications.delete(e.id);
                return t && this.total--, t
            }

            updateWithJson(e) {
                this.cursor = e.cursor, this.total = e.total
            }
        }

        X([d.q], Z.prototype, "cursor", void 0), X([d.q], Z.prototype, "displayOrder", void 0), X([d.q], Z.prototype, "isDeleting", void 0), X([d.q], Z.prototype, "isLoading", void 0), X([d.q], Z.prototype, "isMarkingAsRead", void 0), X([d.q], Z.prototype, "notifications", void 0), X([d.q], Z.prototype, "total", void 0), X([d.q], Z.prototype, "lastNotification", void 0), X([d.h], Z.prototype, "canMarkAsRead", null), X([d.h], Z.prototype, "first", null), X([d.h], Z.prototype, "hasMore", null), X([d.h], Z.prototype, "hasVisibleNotifications", null), X([d.h], Z.prototype, "id", null), X([d.h], Z.prototype, "isSingle", null), X([d.h], Z.prototype, "orderedNotifications", null), X([d.f], Z.prototype, "add", null), X([d.f], Z.prototype, "delete", null), X([d.f], Z.prototype, "deleteItem", null), X([d.f], Z.prototype, "loadMore", null), X([d.f], Z.prototype, "markAsRead", null), X([d.f], Z.prototype, "markStackAsRead", null), X([d.f], Z.prototype, "remove", null), X([d.f], Z.prototype, "updateWithJson", null);
        var Q = s("JlDh"), Y = s("kiUL"), ee = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let te = class {
            constructor(e) {
                Object.defineProperty(this, "notificationStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "legacyPm", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new K.a
                }), Object.defineProperty(this, "types", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object.defineProperty(this, "deletedStacks", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Set
                }), Object.defineProperty(this, "resolver", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Y.a
                }), Object.defineProperty(this, "removeByNotification", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        var t, s;
                        if (null == e.id) return;
                        const r = this.getStack(e), n = this.getOrCreateType(e),
                            i = null == r ? void 0 : r.notifications.get(e.id);
                        null == i ? null != r && e.id < (null !== (s = null === (t = r.cursor) || void 0 === t ? void 0 : t.id) && void 0 !== s ? s : 0) && (this.total--, r.total--, n.total--) : (null == r || r.remove(i), n.total--, this.total--)
                    }
                }), this.getOrCreateType({objectType: null}), Object(d.p)(this)
            }

            get allStacks() {
                return this.allType.stacks
            }

            get allType() {
                return this.getOrCreateType({objectType: null})
            }

            get isEmpty() {
                return 1 === this.types.size && 0 === this.allStacks.size
            }

            get total() {
                return this.allType.total
            }

            set total(e) {
                this.allType.total = e
            }

            flushStore() {
                this.types.clear()
            }

            getOrCreateType(e) {
                let t = this.types.get(e.objectType);
                return null == t && (t = new Q.a(e.objectType, this.resolver), this.types.set(t.name, t)), t
            }

            getStack(e) {
                var t;
                return null === (t = this.types.get(e.objectType)) || void 0 === t ? void 0 : t.stacks.get(Object(W.c)(e))
            }

            handleDispatchAction(e) {
                e instanceof V.a ? this.handleNotificationEventDelete(e) : e instanceof V.c ? this.handleNotificationEventNew(e) : e instanceof V.b ? this.handleNotificationEventMoreLoaded(e) : e instanceof V.d ? this.handleNotificationEventRead(e) : e instanceof O.a && this.flushStore()
            }

            handleNotificationEventDelete(e) {
                this.removeByEvent(e)
            }

            handleNotificationEventMoreLoaded(e) {
                e.context.isWidget || this.updateWithBundle(e.data)
            }

            handleNotificationEventNew(e) {
                const t = e.data;
                let s = this.notificationStore.get(t.id);
                null == s ? (s = z.a.fromJson(t), this.notificationStore.add(s)) : s.updateFromJson(t);
                const r = s.identity, n = this.getOrCreateType(r);
                let i = this.getStack(r);
                null == i && (i = new Z(t.object_id, t.object_type, Object(J.a)(t.name), this.resolver)), i.notifications.has(s.id) || (i.total++, n.total++, this.total++), i.add(s), n.stacks.set(i.id, i), this.allStacks.set(i.id, i)
            }

            handleNotificationEventRead(e) {
            }

            orderedStacksOfType(e) {
                return this.stacksOfType(e).sort((e, t) => t.displayOrder - e.displayOrder)
            }

            removeByEvent(e) {
                for (const t of e.data) {
                    switch (Object(W.b)(t)) {
                        case"type":
                            this.removeByType(t);
                            break;
                        case"stack":
                            this.removeByStack(t, e.readCount);
                            break;
                        case"notification":
                            this.removeByNotification(t)
                    }
                }
            }

            stacksOfType(e) {
                var t, s;
                const r = this.types.get(e), n = [];
                if (null == r) return n;
                const i = null !== (s = null === (t = r.cursor) || void 0 === t ? void 0 : t.id) && void 0 !== s ? s : 0;
                for (const [, a] of r.stacks) void 0 !== (null == r ? void 0 : r.cursor) && a.displayOrder >= i && n.push(a);
                return n
            }

            updateWithBundle(e) {
                var t, s, r;
                null === (t = e.types) || void 0 === t || t.forEach(e => this.updateWithTypeJson(e)), null === (s = e.stacks) || void 0 === s || s.forEach(e => this.updateWithStackJson(e)), null === (r = e.notifications) || void 0 === r || r.forEach(e => this.updateWithNotificationJson(e))
            }

            removeByStack(e, t) {
                const s = this.getStack(e), r = Object(W.c)(e);
                null != s ? (this.deletedStacks.add(r), this.allStacks.delete(r), this.total -= s.total, this.getOrCreateType(e).removeStack(s)) : this.deletedStacks.has(r) || (this.deletedStacks.add(r), this.total -= t)
            }

            removeByType(e) {
                const t = this.getOrCreateType(e);
                if (null === t.name) {
                    for (const [e, t] of this.types) null !== e && this.removeType(t);
                    t.total = 0
                } else this.removeType(t)
            }

            removeType(e) {
                e.stacks.forEach(e => {
                    this.allType.removeStack(e)
                }), e.total = 0, this.types.delete(e.name)
            }

            updateWithNotificationJson(e) {
                var t;
                let s = this.notificationStore.get(e.id);
                null == s ? (s = z.a.fromJson(e), this.notificationStore.add(s)) : s.updateFromJson(e), null === (t = this.getOrCreateType(s.identity).stacks.get(s.stackId)) || void 0 === t || t.add(s)
            }

            updateWithStackJson(e) {
                const t = this.getOrCreateType(Object(W.a)(e));
                let s = t.stacks.get(function (e) {
                    return `${e.object_type}-${e.object_id}-${Object(J.a)(e.name)}`
                }(e));
                null == s ? (s = Z.fromJson(e, this.resolver), t.stacks.set(s.id, s), this.allType.stacks.set(s.id, s)) : s.updateWithJson(e)
            }

            updateWithTypeJson(e) {
                this.getOrCreateType({objectType: e.name}).updateWithJson(e)
            }
        };
        ee([d.q], te.prototype, "legacyPm", void 0), ee([d.q], te.prototype, "types", void 0), ee([d.h], te.prototype, "total", null), ee([d.f], te.prototype, "flushStore", null), ee([d.f], te.prototype, "getOrCreateType", null), ee([d.f], te.prototype, "handleDispatchAction", null), ee([d.f], te.prototype, "handleNotificationEventDelete", null), ee([d.f], te.prototype, "handleNotificationEventMoreLoaded", null), ee([d.f], te.prototype, "handleNotificationEventNew", null), ee([d.f], te.prototype, "handleNotificationEventRead", null), ee([d.f], te.prototype, "removeByEvent", null), ee([d.f], te.prototype, "updateWithBundle", null);
        var se = te = ee([l.b], te), re = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let ne = class extends se {
            handleNotificationEventMoreLoaded(e) {
                e.context.isWidget && this.updateWithBundle(e.data)
            }

            handleNotificationEventNew(e) {
                e.data.is_read || super.handleNotificationEventNew(e)
            }

            handleNotificationEventRead(e) {
                this.removeByEvent(e)
            }

            updateWithBundle(e) {
                super.updateWithBundle(e), null != e.unread_count && (this.total = e.unread_count)
            }
        };
        re([d.s], ne.prototype, "handleNotificationEventMoreLoaded", null), re([d.s], ne.prototype, "handleNotificationEventNew", null), re([d.s], ne.prototype, "handleNotificationEventRead", null), re([d.s], ne.prototype, "updateWithBundle", null);
        var ie = ne = re([l.b], ne), ae = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let oe = class {
            constructor() {
                Object.defineProperty(this, "notifications", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object.defineProperty(this, "stacks", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new se(this)
                }), Object.defineProperty(this, "unreadStacks", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new ie(this)
                }), Object(d.p)(this)
            }

            add(e) {
                this.notifications.set(e.id, e)
            }

            flushStore() {
                this.notifications.clear()
            }

            get(e) {
                return this.notifications.get(e)
            }

            handleDispatchAction(e) {
                e instanceof V.a ? this.handleNotificationEventDelete(e) : e instanceof V.d ? this.handleNotificationEventRead(e) : e instanceof O.a && this.flushStore()
            }

            handleNotificationEventDelete(e) {
                this.eachByEvent(e, e => {
                    this.notifications.delete(e.id)
                })
            }

            handleNotificationEventRead(e) {
                this.eachByEvent(e, e => {
                    e.isRead = !0
                })
            }

            eachByEvent(e, t) {
                for (const s of e.data) {
                    switch (Object(W.b)(s)) {
                        case"type":
                            this.eachByType(s, t);
                            break;
                        case"stack":
                            this.eachByStack(s, t);
                            break;
                        case"notification": {
                            if (null == s.id) return;
                            const e = this.get(s.id);
                            null != e && t(e);
                            break
                        }
                    }
                }
            }

            eachByStack(e, t) {
                const s = Object(W.c)(e);
                this.notifications.forEach(e => {
                    e.stackId === s && t(e)
                })
            }

            eachByType(e, t) {
                this.notifications.forEach(s => {
                    null != e.objectType && s.objectType !== e.objectType || t(s)
                })
            }
        };
        ae([d.q], oe.prototype, "notifications", void 0), ae([d.f], oe.prototype, "add", null), ae([d.f], oe.prototype, "flushStore", null), ae([d.f], oe.prototype, "handleDispatchAction", null), ae([d.f], oe.prototype, "handleNotificationEventDelete", null), ae([d.f], oe.prototype, "handleNotificationEventRead", null);
        var le = oe = ae([l.b], oe), ce = s("IfWv"), ue = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class de {
            constructor() {
                Object.defineProperty(this, "clients", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object(d.p)(this)
            }

            handleDispatchAction(e) {
                e instanceof O.a && this.flushStore()
            }

            initialize(e) {
                for (const t of e) this.updateWithJson(t)
            }

            updateWithJson(e) {
                const t = new ce.a(e);
                return this.clients.set(t.id, t), t
            }

            flushStore() {
                this.clients.clear()
            }
        }

        ue([d.q], de.prototype, "clients", void 0), ue([d.f], de.prototype, "updateWithJson", null), ue([d.f], de.prototype, "flushStore", null);
        var pe = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        const me = {
            currentSort: "new",
            hasMoreComments: {},
            loadingFollow: null,
            loadingSort: null,
            pinnedCommentIds: [],
            topLevelCommentIds: [],
            topLevelCount: 0,
            total: 0,
            userFollow: !1
        };

        class he {
            constructor(e) {
                Object.defineProperty(this, "commentStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "account", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {client: null, isCreatingNewClient: !1, newClientVisible: !1}
                }), Object.defineProperty(this, "comments", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: Object.assign({}, me)
                }), Object.defineProperty(this, "orderedCommentsByParentId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {}
                }), Object(d.p)(this)
            }

            exportCommentsUIState() {
                return {comments: this.comments, orderedCommentsByParentId: this.orderedCommentsByParentId}
            }

            getOrderedCommentsByParentId(e) {
                return this.populateOrderedCommentsForParentId(e), this.orderedCommentsByParentId[e]
            }

            importCommentsUIState(e) {
                this.comments = Object.assign({}, me, e.comments), this.orderedCommentsByParentId = e.orderedCommentsByParentId
            }

            initializeWithCommentBundleJson(e) {
                this.comments.hasMoreComments = {}, this.comments.hasMoreComments[e.has_more_id] = e.has_more, this.comments.currentSort = e.sort, this.comments.userFollow = e.user_follow, this.comments.topLevelCount = e.top_level_count ? e.top_level_count : 0, this.comments.total = e.total ? e.total : 0, null != e.comments && (this.comments.topLevelCommentIds = e.comments.map(e => e.id)), this.updatePinnedCommentIds(e), this.orderedCommentsByParentId = {}
            }

            updateFromCommentUpdated(e) {
                this.updatePinnedCommentIds(e)
            }

            updateFromCommentsAdded(e) {
                this.comments.hasMoreComments[e.has_more_id] = e.has_more, e.top_level_count && e.total && (this.comments.topLevelCount = e.top_level_count, this.comments.total = e.total);
                for (const t of e.comments) {
                    const e = new L(t), s = e.parentId;
                    null == s ? this.comments.topLevelCommentIds.push(e.id) : (this.populateOrderedCommentsForParentId(s), this.orderedCommentsByParentId[s].push(e))
                }
            }

            updateFromCommentsNew(e) {
                e.top_level_count && e.total && (this.comments.topLevelCount = e.top_level_count, this.comments.total = e.total);
                const t = new L(e.comments[0]), s = t.parentId;
                null == s ? this.comments.topLevelCommentIds.unshift(t.id) : (this.populateOrderedCommentsForParentId(s), this.orderedCommentsByParentId[s].unshift(t))
            }

            orderComments(e) {
                switch (this.comments.currentSort) {
                    case"old":
                        return Object(u.orderBy)(e, "createdAt", "asc");
                    case"top":
                        return Object(u.orderBy)(e, "votesCount", "desc");
                    default:
                        return Object(u.orderBy)(e, "createdAt", "desc")
                }
            }

            populateOrderedCommentsForParentId(e) {
                if (null == this.orderedCommentsByParentId[e]) {
                    const t = this.commentStore.getRepliesByParentId(e);
                    this.orderedCommentsByParentId[e] = this.orderComments(t)
                }
            }

            updatePinnedCommentIds(e) {
                null != e.pinned_comments && (this.comments.pinnedCommentIds = e.pinned_comments.map(e => e.id))
            }
        }

        pe([d.q], he.prototype, "account", void 0), pe([d.q], he.prototype, "comments", void 0), pe([d.f], he.prototype, "importCommentsUIState", null), pe([d.f], he.prototype, "initializeWithCommentBundleJson", null), pe([d.f], he.prototype, "updateFromCommentUpdated", null), pe([d.f], he.prototype, "updateFromCommentsAdded", null), pe([d.f], he.prototype, "updateFromCommentsNew", null);
        var be = s("srn7"), fe = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        let ve = class {
            constructor() {
                Object.defineProperty(this, "users", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: d.q.map()
                }), Object(d.p)(this)
            }

            flushStore() {
                this.users = d.q.map()
            }

            get(e) {
                return this.users.get(e)
            }

            getOrCreate(e, t) {
                let s = this.users.get(e);
                return s && s.loaded ? s : (s = t ? be.a.fromJson(t) : new be.a(e), this.users.set(e, s), s.loaded || s.load(), s)
            }

            handleDispatchAction(e) {
                e instanceof k.a && this.updateWithJson(e.json.users)
            }

            updateWithJson(e) {
                if (null != e) for (const t of e) {
                    const e = be.a.fromJson(t);
                    this.users.set(e.id, e)
                }
            }
        };
        fe([d.q], ve.prototype, "users", void 0), fe([d.f], ve.prototype, "flushStore", null), fe([d.f], ve.prototype, "getOrCreate", null), fe([d.f], ve.prototype, "updateWithJson", null);
        var ge = ve = fe([l.b], ve);

        class ye {
            constructor() {
                Object.defineProperty(this, "beatmapsetSearch", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "beatmapsetStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "channelStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "chatState", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "clientStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "commentStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "commentableMetaStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "notificationStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "ownClientStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "uiState", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "userStore", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), this.beatmapsetStore = new E, this.beatmapsetSearch = new r.a(this.beatmapsetStore), this.clientStore = new U, this.commentableMetaStore = new F, this.commentStore = new q, this.notificationStore = new le, this.ownClientStore = new de, this.userStore = new ge, this.channelStore = new I, this.chatState = new _(this.channelStore), this.uiState = new he(this.commentStore), Object(d.p)(this)
            }

            updateWithCommentBundleJson(e) {
                this.commentableMetaStore.updateWithJson(e.commentable_meta), this.commentStore.updateWithJson(e.comments), this.commentStore.updateWithJson(e.included_comments), this.commentStore.updateWithJson(e.pinned_comments), this.userStore.updateWithJson(e.users), this.commentStore.addVoted(e.user_votes)
            }
        }

        (function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            i > 3 && a && Object.defineProperty(t, s, a)
        })([d.f], ye.prototype, "updateWithCommentBundleJson", null)
    }, VxeA: function (e, t, s) {
        "use strict";
        const r = Object.freeze({
            colour: "hsl(var(--hsl-l1))",
            has_listing: !1,
            has_playmodes: !1,
            id: -1,
            identifier: "owner",
            is_probationary: !1,
            name: "Beatmap Mapper",
            short_name: "MAPPER"
        });
        t.a = r
    }, WD1s: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("lv9K"), n = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };
        const i = ["extra", "general", "genre", "language", "mode", "nsfw", "played", "query", "rank", "sort", "status"];

        class a {
            constructor(e) {
                var t;
                Object.defineProperty(this, "extra", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "general", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "genre", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "language", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "mode", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "nsfw", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "played", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "query", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "rank", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "sort", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "status", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                });
                const s = BeatmapsetFilter.filtersFromUrl(e);
                for (const r of i) this[r] = null !== (t = s[r]) && void 0 !== t ? t : null;
                Object(r.p)(this), Object(r.l)(this, "query", e => {
                    var t;
                    return e.newValue = osu.presence(null === (t = e.newValue) || void 0 === t ? void 0 : t.trim()), e
                })
            }

            get displaySort() {
                return this.selectedValue("sort")
            }

            get queryParams() {
                const e = this.values;
                return BeatmapsetFilter.queryParamsFromFilters(e)
            }

            get searchSort() {
                const [e, t] = this.displaySort.split("_");
                return {field: e, order: t}
            }

            selectedValue(e) {
                const t = this[e];
                return null == t ? BeatmapsetFilter.getDefault(this.values, e) : t
            }

            toKeyString() {
                const e = this.values, t = BeatmapsetFilter.fillDefaults(e), s = [];
                for (const r of i) s.push(`${r}=${t[r]}`);
                return s.join("&")
            }

            update(e) {
                (void 0 !== e.query && e.query !== this.query || void 0 !== e.status && e.status !== this.status) && (this.sort = null);
                for (const t of i) {
                    const s = e[t];
                    void 0 !== s && (this[t] = s)
                }
            }

            get values() {
                return Object.assign({}, this)
            }
        }

        n([r.q], a.prototype, "extra", void 0), n([r.q], a.prototype, "general", void 0), n([r.q], a.prototype, "genre", void 0), n([r.q], a.prototype, "language", void 0), n([r.q], a.prototype, "mode", void 0), n([r.q], a.prototype, "nsfw", void 0), n([r.q], a.prototype, "played", void 0), n([r.q], a.prototype, "query", void 0), n([r.q], a.prototype, "rank", void 0), n([r.q], a.prototype, "sort", void 0), n([r.q], a.prototype, "status", void 0), n([r.h], a.prototype, "displaySort", null), n([r.h], a.prototype, "queryParams", null), n([r.h], a.prototype, "searchSort", null), n([r.f], a.prototype, "update", null), n([r.h], a.prototype, "values", null)
    }, "WJQ/": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("Hs9Z"), n = s("/G9H"), i = s("tX/w");

            class a extends n.Component {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "count", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (null != this.props.counts) return this.props.counts[e];
                            const t = Object(r.sumBy)(this.props.beatmaps.get(e), e => e.convert ? 0 : 1);
                            return t > 0 ? t : void 0
                        }
                    }), Object.defineProperty(this, "switchMode", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            t.preventDefault();
                            const s = t.currentTarget, r = s.dataset.mode;
                            this.props.currentMode !== r && "true" !== s.dataset.disabled && e.publish("playmode:set", {mode: r})
                        }
                    })
                }

                render() {
                    return n.createElement("div", {className: "game-mode game-mode--beatmapsets"}, n.createElement("ul", {className: "game-mode__items"}, [...this.props.beatmaps].map(([e, t]) => {
                        var s, r, a;
                        const o = 0 === t.length,
                            l = Object(i.a)("game-mode-link", {active: e === this.props.currentMode, disabled: o}),
                            c = this.count(e);
                        return n.createElement("li", {
                            key: e,
                            className: "game-mode__item"
                        }, n.createElement("a", {
                            className: l,
                            "data-disabled": o.toString(),
                            "data-mode": e,
                            href: null !== (a = null === (r = (s = this.props).hrefFunc) || void 0 === r ? void 0 : r.call(s, e)) && void 0 !== a ? a : "#",
                            onClick: this.switchMode
                        }, osu.trans(`beatmaps.mode.${e}`), null != c && n.createElement("span", {className: "game-mode-link__badge"}, c)))
                    })))
                }
            }
        }).call(this, s("5wds"))
    }, WLnA: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        })), s.d(t, "b", (function () {
            return i
        }));
        const r = new class {
            constructor() {
                Object.defineProperty(this, "listeners", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Set
                }), Object.defineProperty(this, "trace", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "dispatch", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        this.trace && console.debug("Dispatcher::dispatch", e), this.listeners.forEach(t => {
                            t.handleDispatchAction(e)
                        })
                    }
                })
            }

            get size() {
                return this.listeners.size
            }

            clear() {
                this.listeners.clear()
            }

            has(e) {
                return this.listeners.has(e)
            }

            register(e) {
                this.listeners.add(e)
            }

            unregister(e) {
                this.listeners.delete(e)
            }
        };
        const n = r.dispatch;

        function i(e) {
            return class extends e {
                constructor(...e) {
                    var t;
                    super(...e), "object" == typeof (t = this) && null != t && "handleDispatchAction" in t && r.register(this)
                }
            }
        }
    }, XFVs: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return l
        }));
        var r = s("55pz"), n = s("0h6b"), i = s("7EfK"), a = s("/G9H"), o = s("tX/w");

        function l({modifiers: e, post: t}) {
            let s;
            null != t.first_image && (s = a.createElement("img", {className: "news-card__cover", src: t.first_image}));
            let l = t.preview;
            return null == l && (l = ""), a.createElement("a", {
                className: Object(o.a)("news-card", null != e ? e : ["index", "hover"]),
                href: Object(n.a)("news.show", {news: t.slug})
            }, a.createElement("div", {className: "news-card__cover-container"}, s, a.createElement("div", {
                className: "news-card__time js-tooltip-time",
                title: t.published_at
            }, i.utc(t.published_at).format("ll"))), a.createElement("div", {className: "news-card__main"}, a.createElement("div", {className: "news-card__row news-card__row--title"}, t.title), a.createElement("div", {
                dangerouslySetInnerHTML: {__html: l},
                className: "news-card__row news-card__row--preview"
            }), a.createElement("div", {className: "news-card__row news-card__row--author"}, a.createElement(r.a, {
                mappings: {user: a.createElement("strong", null, t.author)},
                pattern: osu.trans("news.show.by")
            }))))
        }
    }, XMXc: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return f
        }));
        var r = s("/G9H"), n = s("KHX3"), i = s("iH//"), a = s("vZz4");

        function o() {
            function e(e, t, s) {
                const r = new RegExp(a.j).exec(t);
                if (!r || 0 !== r.index) return;
                if (s) return !0;
                const [n, , i] = r;
                return e(n)({children: [{type: "text", value: i}], type: "link", url: n})
            }

            const t = this.Parser, s = t.prototype.inlineTokenizers, r = t.prototype.inlineMethods;
            e.locator = function (e, t) {
                return e.indexOf("scrape", t)
            }, s.link = e, r.unshift("link")
        }

        var l = s("Cy5r"), c = s("sHNI"), u = s("UxfW"), d = s("tX/w"), p = s("Y9Pv"), m = s("DHbW");
        const h = ({data: e}) => {
            const t = "beatmap-discussion-review-post-embed-preview", s = r.useContext(m.a), n = r.useContext(p.a),
                i = s[e.discussion_id];
            if (!i) return r.createElement("div", {className: Object(d.a)(t, ["deleted", "lighter"])}, r.createElement("div", {className: `${t}__missing`}, osu.trans("beatmaps.discussions.review.embed.missing")));
            const a = null == i.beatmap_id ? void 0 : n[i.beatmap_id], o = ["lighter"];
            "praise" === i.message_type ? o.push("praise") : i.resolved && o.push("resolved"), null !== i.beatmap_id || o.push("general-all"), i.deleted_at && o.push("deleted");
            const l = () => {
                if (i.parent_id) return r.createElement("div", {className: `${t}__link`}, r.createElement("a", {
                    className: `${t}__link-text js-beatmap-discussion--jump`,
                    href: BeatmapDiscussionHelper.url({discussion: i}),
                    title: osu.trans("beatmap_discussions.review.go_to_child")
                }, r.createElement("i", {className: "fas fa-external-link-alt"})))
            };
            return r.createElement("div", {className: Object(d.a)(t, o)}, r.createElement("div", {className: `${t}__content`}, r.createElement("div", {className: `${t}__selectors`}, r.createElement("div", {className: "icon-dropdown-menu icon-dropdown-menu--disabled"}, null != a && r.createElement(u.a, {
                beatmap: a,
                withTooltip: !0
            }), !i.beatmap_id && r.createElement("i", {
                className: "fas fa-fw fa-star-of-life",
                title: osu.trans("beatmaps.discussions.mode.scopes.generalAll")
            })), r.createElement("div", {className: "icon-dropdown-menu icon-dropdown-menu--disabled"}, (() => {
                const e = i.message_type;
                return r.createElement("div", {className: `beatmap-discussion-message-type beatmap-discussion-message-type--${e}`}, r.createElement("i", {
                    className: c.a[e],
                    title: osu.trans(`beatmaps.discussions.message_type.${e}`)
                }))
            })()), r.createElement("div", {className: `${t}__timestamp`}, (() => r.createElement("div", {className: `${t}__timestamp-text`}, null !== i.timestamp ? BeatmapDiscussionHelper.formatTimestamp(i.timestamp) : osu.trans("beatmap_discussions.timestamp_display.general")))()), r.createElement("div", {className: `${t}__stripe`}), l()), r.createElement("div", {className: `${t}__stripe`}), r.createElement("div", {className: `${t}__message-container`}, r.createElement("div", {
                className: `${t}__body`,
                dangerouslySetInnerHTML: {__html: BeatmapDiscussionHelper.format((i.starting_post || i.posts[0]).message)}
            })), l()))
        };

        function b() {
            function e(e, t, s) {
                const r = new RegExp(/^((\d{2,}:[0-5]\d[:.]\d{3})( \((?:\d[,|])*\d\))?)/).exec(t);
                if (!r) return;
                if (s) return !0;
                const [n, i] = r;
                return e(n)({children: [{type: "text", value: i}], href: Object(a.h)(i), type: "timestamp"})
            }

            const t = this.Parser, s = t.prototype.inlineTokenizers, r = t.prototype.inlineMethods;
            e.locator = function (e, t) {
                const s = e.substr(t).search(/[0-9]{2}:/);
                return s < 0 ? s : s + t
            }, s.timestamp = e, r.unshift("timestamp")
        }

        class f extends r.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "linkRenderer", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        const t = Object(i.c)(e.href);
                        return r.createElement("a", Object.assign({}, e, t))
                    }
                })
            }

            embed(e) {
                return r.createElement("div", {
                    key: osu.uuid(),
                    className: "beatmap-discussion-review-post__block"
                }, r.createElement(h, {data: {discussion_id: e}}))
            }

            paragraph(e) {
                return r.createElement(n, {
                    key: osu.uuid(),
                    plugins: [[l.a, {allowedBlocks: ["paragraph"], allowedInlines: ["emphasis", "strong"]}], o, b],
                    renderers: {
                        link: this.linkRenderer,
                        paragraph: e => r.createElement("div", {className: "beatmap-discussion-review-post__block"}, r.createElement("div", Object.assign({className: "beatmapset-discussion-message"}, e))),
                        timestamp: e => r.createElement("a", Object.assign({className: "beatmap-discussion-timestamp-decoration"}, e))
                    },
                    source: e,
                    unwrapDisallowed: !0
                })
            }

            render() {
                const e = [];
                try {
                    JSON.parse(this.props.message).forEach(t => {
                        switch (t.type) {
                            case"paragraph": {
                                const s = "" === t.text.trim() ? "&nbsp;" : t.text;
                                e.push(this.paragraph(s));
                                break
                            }
                            case"embed":
                                t.discussion_id && e.push(this.embed(t.discussion_id))
                        }
                    })
                } catch (t) {
                    e.push(r.createElement("div", {key: osu.uuid()}, "[error parsing review]"))
                }
                return r.createElement("div", {className: "beatmap-discussion-review-post"}, e)
            }
        }
    }, XMjx: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("SifC"), n = s("GhGe"), i = s("Cy5r");

        function a(e, t) {
            let s;
            try {
                s = JSON.parse(e)
            } catch (c) {
                return console.error("error parsing srcDoc"), []
            }
            const a = n().use(r).use(i.a, {allowedBlocks: ["paragraph"], allowedInlines: ["emphasis", "strong"]}),
                l = [];
            return s.forEach(e => {
                switch (e.type) {
                    case"paragraph":
                        if (osu.present(e.text.trim())) {
                            const t = a.parse(e.text);
                            if (!t.children || t.children.length < 1) {
                                console.error("children missing... ?");
                                break
                            }
                            l.push({children: o(t.children), type: "paragraph"})
                        } else l.push({children: [{text: ""}], type: "paragraph"});
                        break;
                    case"embed": {
                        const s = e, r = t[s.discussion_id];
                        if (null == r) {
                            console.error("unknown/external discussion referenced", s.discussion_id);
                            break
                        }
                        l.push({
                            beatmapId: r.beatmap_id,
                            children: [{text: (r.starting_post || r.posts[0]).message}],
                            discussionId: r.id,
                            discussionType: r.message_type,
                            timestamp: BeatmapDiscussionHelper.formatTimestamp(r.timestamp),
                            type: "embed"
                        });
                        break
                    }
                    default:
                        console.error("unknown block encountered", e)
                }
            }), l
        }

        function o(e, t) {
            let s = [];
            const r = null != t ? t : {bold: !1, italic: !1};
            return e ? (e.forEach(e => {
                const t = {bold: r.bold || "strong" === e.type, italic: r.italic || "emphasis" === e.type};
                if (Array.isArray(e.children)) s = s.concat(o(e.children, t)); else {
                    const r = {text: e.value || ""};
                    t.bold && (r.bold = !0), t.italic && (r.italic = !0), s.push(r)
                }
            }), s) : [{text: ""}]
        }
    }, XvDi: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
            constructor(e) {
                Object.defineProperty(this, "message", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, Y9Pv: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = r.createContext({})
    }, YvoO: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("bpgk"), n = s("WLnA"), i = s("0h6b");

            class a {
                constructor() {
                    Object.defineProperty(this, "handleUserLogin", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            Object(n.a)(new r.a)
                        }
                    }), Object.defineProperty(this, "handleUserLogout", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = !!e.currentTarget.dataset.redirectHome;
                            this.logout(t)
                        }
                    }), e(document).on("ajax:success", ".js-logout-link", this.handleUserLogout).on("ajax:success", ".js-login-form", this.handleUserLogin)
                }

                logout(e = !1) {
                    localStorage.clear(), e ? location.href = Object(i.a)("home") : location.reload()
                }
            }
        }).call(this, s("5wds"))
    }, ZG74: function (e, t, s) {
        "use strict";
        s.d(t, "c", (function () {
            return n
        })), s.d(t, "a", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        }));
        var r = s("/G9H");

        function n(e) {
            this.setState({activeKey: e})
        }

        const i = Object(r.createContext)({
            activeKeyDidChange: e => {
            }
        }), a = Object(r.createContext)(null)
    }, Zp7Q: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return u
        }));
        var r = s("UBw1"), n = s("0h6b"), i = s("Hs9Z"), a = s("srn7"), o = s("/G9H"), l = s("tX/w"), c = s("vZz4");

        class u extends o.PureComponent {
            get beatmapsetId() {
                var e;
                return null === (e = this.props.event.beatmapset) || void 0 === e ? void 0 : e.id
            }

            get discussionId() {
                const e = this.props.event.comment;
                if (null != e && "object" == typeof e && "beatmap_discussion_id" in e) return e.beatmap_discussion_id
            }

            get discussion() {
                var e, t, s;
                return null !== (e = this.props.event.discussion) && void 0 !== e ? e : null === (t = this.props.discussions) || void 0 === t ? void 0 : t[null !== (s = this.discussionId) && void 0 !== s ? s : ""]
            }

            get firstPost() {
                var e, t, s, r;
                return null !== (t = null === (e = this.discussion) || void 0 === e ? void 0 : e.starting_post) && void 0 !== t ? t : null === (r = null === (s = this.discussion) || void 0 === s ? void 0 : s.posts) || void 0 === r ? void 0 : r[0]
            }

            render() {
                return "discussions" === this.props.mode ? this.renderDiscussionsVersion() : this.renderProfileVersion()
            }

            renderDiscussionsVersion() {
                var e;
                const t = null !== (e = this.props.time) && void 0 !== e ? e : this.props.event.created_at;
                return o.createElement("div", {className: "beatmapset-event"}, o.createElement("div", {
                    className: "beatmapset-event__icon",
                    style: this.iconStyle()
                }), o.createElement("div", {className: "beatmapset-event__time"}, o.createElement(r.a, {
                    dateTime: t,
                    format: "LT"
                })), o.createElement("div", {
                    className: "beatmapset-event__content",
                    dangerouslySetInnerHTML: {__html: this.contentText()}
                }))
            }

            renderProfileVersion() {
                var e, t;
                const s = null === (t = null === (e = this.props.event.beatmapset) || void 0 === e ? void 0 : e.covers) || void 0 === t ? void 0 : t.list;
                let i;
                return null != this.beatmapsetId && (i = Object(n.a)("beatmapsets.discussion", {beatmapset: this.beatmapsetId}), null != this.discussionId && (i = `${i}#/${this.discussionId}`)), o.createElement("div", {className: "beatmapset-event"}, null != i ? o.createElement("a", {href: i}, o.createElement("img", {
                    className: "beatmapset-cover",
                    src: s
                })) : o.createElement("span", {className: "beatmapset-cover"}, "beatmap deleted"), o.createElement("div", {
                    className: Object(l.a)("beatmapset-event__icon", ["beatmapset-activities"]),
                    style: this.iconStyle()
                }), o.createElement("div", null, o.createElement("div", {
                    className: "beatmapset-event__content",
                    dangerouslySetInnerHTML: {__html: this.contentText()}
                }), o.createElement("div", {className: "beatmap-discussion-post__info"}, o.createElement(r.a, {
                    dateTime: this.props.event.created_at,
                    relative: !0
                }))))
            }

            contentText() {
                var e, t, s;
                let r, o = "", l = "[unknown user]", u = "", d = "";
                if (null != this.discussionId) {
                    if (null == this.discussion) d = Object(n.a)("beatmapsets.discussions.show", {discussion: this.discussionId}), u = osu.trans("beatmapset_events.item.discussion_deleted"); else {
                        const t = null === (e = this.firstPost) || void 0 === e ? void 0 : e.message;
                        d = BeatmapDiscussionHelper.url({discussion: this.discussion}), u = null != t ? BeatmapDiscussionHelper.previewMessage(t) : "[no preview]";
                        const s = this.props.users[this.discussion.user_id];
                        null != s && (l = Object(c.f)(Object(n.a)("users.show", {user: s.id}), s.username))
                    }
                    o = Object(c.f)(d, `#${this.discussionId}`, {classNames: ["js-beatmap-discussion--jump"]})
                } else "string" == typeof this.props.event.comment && (u = BeatmapDiscussionHelper.format(this.props.event.comment, {newlines: !1}));
                if ("discussion_lock" !== this.props.event.type && "remove_from_loved" !== this.props.event.type || (u = BeatmapDiscussionHelper.format(this.props.event.comment.reason, {newlines: !1})), null != this.props.event.user_id) {
                    const e = this.props.users[this.props.event.user_id];
                    r = null == e ? Object(i.escape)(a.b.username) : Object(c.f)(Object(n.a)("users.show", {user: e.id}), e.username)
                }
                const p = {discussion: o, discussion_user: l, text: u, user: r};
                if (null != this.props.event.comment && "object" == typeof this.props.event.comment) for (const [n, i] of Object.entries(this.props.event.comment)) "number" != typeof i && "string" != typeof i || (p[n] = i);
                let m = this.props.event.type;
                switch (this.props.event.type) {
                    case"disqualify":
                        "string" == typeof this.props.event.comment && (m = "disqualify_legacy");
                        break;
                    case"nominate": {
                        const e = null === (t = this.props.event.comment) || void 0 === t ? void 0 : t.modes;
                        if (null != e && e.length > 0) {
                            m = "nominate_modes";
                            const t = e.map(e => osu.trans(`beatmaps.mode.${e}`));
                            p.modes = osu.transArray(t)
                        }
                        break
                    }
                    case"nsfw_toggle":
                        m += `.${(null === (s = this.props.event.comment) || void 0 === s ? void 0 : s.new) ? "to_1" : "to_0"}`;
                        break;
                    case"beatmap_owner_change": {
                        const e = this.props.event.comment;
                        p.new_user = Object(c.f)(Object(n.a)("users.show", {user: e.new_user_id}), e.new_user_username), p.beatmap = Object(c.f)(Object(n.a)("beatmaps.show", {beatmap: e.beatmap_id}), e.beatmap_version);
                        break
                    }
                    case"nomination_reset_received": {
                        const e = this.props.event.comment;
                        "profile" === this.props.mode ? (m += "_profile", p.user = Object(c.f)(Object(n.a)("users.show", {user: e.source_user_id}), e.source_user_username)) : p.source_user = Object(c.f)(Object(n.a)("users.show", {user: e.source_user_id}), e.source_user_username)
                    }
                }
                const h = `beatmapset_events.event.${m}`;
                let b = osu.trans(h, p);
                return null == r || osu.trans(h).includes(":user") || (b += ` (${r})`), b
            }

            iconStyle() {
                return {"--bg": `var(--bg-${Object(i.kebabCase)(this.props.event.type)})`}
            }
        }
    }, aCRl: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r, n = s("/G9H"), i = s("I8Ok");
            n.createElement, r = {add: "fas fa-plus", fix: "fas fa-check", misc: "far fa-circle"};
            var a = function (t) {
                var s, n, a, o, l, c, u;
                return s = t.entry, n = e.escape(s.title).replace(/(`+)([^`]+)\1/g, "<code>$2</code>"), Object(i.div)({
                    className: "changelog-entry",
                    key: s.id
                }, Object(i.div)({className: "changelog-entry__row"}, Object(i.div)({className: "changelog-entry__title " + (s.major ? "changelog-entry__title--major" : "")}, Object(i.span)({className: "changelog-entry__title-icon"}, Object(i.span)({className: "changelog-entry__icon " + r[s.type]})), null != s.url ? Object(i.a)({
                    href: s.url,
                    className: "changelog-entry__link",
                    dangerouslySetInnerHTML: {__html: n}
                }) : Object(i.span)({dangerouslySetInnerHTML: {__html: n}}), null != s.github_url ? Object(i.span)(null, " (", Object(i.a)({
                    className: "changelog-entry__link",
                    href: s.github_url
                }, s.repository.replace(/^.*\//, "") + "#" + s.github_pull_request_id), ")") : void 0, (u = e.escape(s.github_user.display_name), a = null != (c = null != (o = s.github_user.github_url) ? o : s.github_user.user_url) ? "<a data-user-id='" + (null != (l = s.github_user.user_id) ? l : "") + "' class='changelog-entry__user-link js-usercard' href='" + e.escape(c) + "' >" + u + "</a>" : u, Object(i.span)({
                    className: "changelog-entry__user",
                    dangerouslySetInnerHTML: {__html: osu.trans("changelog.entry.by", {user: a})}
                })))), null != s.message_html ? Object(i.div)({
                    className: "changelog-entry__row changelog-entry__row--message",
                    dangerouslySetInnerHTML: {__html: s.message_html}
                }) : void 0)
            }
        }).call(this, s("Hs9Z"))
    }, aMFG: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return r
            }));

            class r {
                constructor() {
                    Object.defineProperty(this, "sitekey", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: ""
                    }), Object.defineProperty(this, "triggered", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "container", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => document.querySelector(".js-captcha--container")
                    }), Object.defineProperty(this, "disableSubmit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = this.submitButton();
                            e && (e.disabled = !0)
                        }
                    }), Object.defineProperty(this, "enableSubmit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = this.submitButton();
                            e && (e.disabled = !1)
                        }
                    }), Object.defineProperty(this, "init", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            this.sitekey = e, this.triggered = t, this.render()
                        }
                    }), Object.defineProperty(this, "isEnabled", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => this.container() && "object" == typeof grecaptcha && "function" == typeof grecaptcha.render && "" !== this.sitekey && this.triggered
                    }), Object.defineProperty(this, "isLoaded", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e;
                            return "" !== (null === (e = this.container()) || void 0 === e ? void 0 : e.innerHTML)
                        }
                    }), Object.defineProperty(this, "render", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.isEnabled() && !this.isLoaded() && (grecaptcha.render(this.container(), {
                                callback: this.enableSubmit,
                                "error-callback": this.disableSubmit,
                                "expired-callback": this.disableSubmit,
                                sitekey: this.sitekey,
                                theme: "dark"
                            }), this.disableSubmit())
                        }
                    }), Object.defineProperty(this, "reset", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.isEnabled() && (grecaptcha.reset(), this.disableSubmit())
                        }
                    }), Object.defineProperty(this, "submitButton", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => document.querySelector(".js-captcha--submit-button")
                    }), Object.defineProperty(this, "trigger", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.triggered || (this.triggered = !0, this.render())
                        }
                    }), Object.defineProperty(this, "untrigger", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.isEnabled() && (this.triggered = !1, this.container().innerHTML = "", this.enableSubmit())
                        }
                    }), e(document).on("turbolinks:load", this.render)
                }
            }
        }).call(this, s("5wds"))
    }, b6Rb: function (e, t, s) {
        "use strict";
        var r = s("7nlf"), n = s("UAat"), i = s("lv9K"), a = s("KUml"), o = s("f4vq"), l = s("/G9H"), c = s("tX/w"),
            u = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };
        let d = class extends l.Component {
            constructor(e) {
                super(e), Object.defineProperty(this, "state", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {forceShow: !1}
                }), Object.defineProperty(this, "handleClick", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: () => {
                        this.setState({forceShow: !this.state.forceShow})
                    }
                }), Object(i.p)(this)
            }

            get isBlocked() {
                return o.a.currentUserModel.blocks.has(this.props.user.id)
            }

            render() {
                let e;
                const t = ["full"];
                return this.isBlocked && !this.state.forceShow && (e = "osu-layout__no-scroll", t.push("masked")), l.createElement("div", {className: e}, this.isBlocked && this.renderBanner(), l.createElement("div", {className: Object(c.a)("osu-layout", t)}, this.props.children))
            }

            renderBanner() {
                const e = l.createElement("div", {className: "grid-items grid-items--notification-banner-buttons"}, l.createElement("div", null, l.createElement(r.a, {userId: this.props.user.id})), l.createElement("div", null, l.createElement("button", {
                    className: "textual-button",
                    onClick: this.handleClick,
                    type: "button"
                }, l.createElement("span", null, l.createElement("i", {className: "textual-button__icon fas fa-low-vision"}), " ", this.state.forceShow ? osu.trans("users.blocks.hide_profile") : osu.trans("users.blocks.show_profile")))));
                return l.createElement("div", {className: "osu-page"}, l.createElement(n.a, {
                    message: e,
                    title: osu.trans("users.blocks.banner_text"),
                    type: "warning"
                }))
            }
        };
        u([i.h], d.prototype, "isBlocked", null), d = u([a.b], d), t.a = d
    }, bPNN: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("fqdx"), n = s("f4vq"), i = s("phBA");

            class a {
                constructor(t) {
                    Object.defineProperty(this, "captcha", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "callback", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "show", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            this.callback = t, window.setTimeout(() => {
                                var t;
                                e(document).trigger("gallery:close"), null === (t = e(".js-user-login--menu")[0]) || void 0 === t || t.click()
                            }, 0)
                        }
                    }), Object.defineProperty(this, "showIfGuest", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => null == n.a.currentUser && (this.show(e), !0)
                    }), Object.defineProperty(this, "showOnError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            var s;
                            return 401 === e.status && "basic" === (null === (s = e.responseJSON) || void 0 === s ? void 0 : s.authentication) && (null != n.a.currentUser ? osu.reloadPage() : this.show(t), !0)
                        }
                    }), Object.defineProperty(this, "clearError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            e(".js-login-form--error").text("")
                        }
                    }), Object.defineProperty(this, "loginError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            t.preventDefault(), t.stopPropagation(), e(".js-login-form--error").text(osu.xhrErrorMessage(s)), window.setTimeout(() => {
                                var e;
                                (null === (e = null == s ? void 0 : s.responseJSON) || void 0 === e ? void 0 : e.captcha_triggered) && this.captcha.trigger(), this.captcha.reset()
                            }, 0)
                        }
                    }), Object.defineProperty(this, "loginSuccess", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            const r = this.callback;
                            this.reset(), this.refreshToken(), e.publish("user:update", s.user), window.setTimeout(() => {
                                var t;
                                null === (t = e(".js-user-login--menu")[0]) || void 0 === t || t.click(), e(".js-user-header").replaceWith(s.header), e(".js-user-header-popup").html(s.header_popup), this.captcha.untrigger(), (null != r ? r : osu.reloadPage)()
                            }, 0)
                        }
                    }), Object.defineProperty(this, "onError", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            this.showOnError(t, Object(i.c)(e.target))
                        }
                    }), Object.defineProperty(this, "refreshToken", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var t;
                            const s = null !== (t = r.get("XSRF-TOKEN")) && void 0 !== t ? t : null;
                            e('[name="_token"]').attr("value", s), e('[name="csrf-token"]').attr("content", s)
                        }
                    }), Object.defineProperty(this, "reset", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.callback = void 0
                        }
                    }), Object.defineProperty(this, "showOnClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            e.preventDefault(), this.show()
                        }
                    }), Object.defineProperty(this, "showOnLoad", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.showLoginModal && (window.showLoginModal = void 0, this.show())
                        }
                    }), Object.defineProperty(this, "showToContinue", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (null != n.a.currentUser) return;
                            e.preventDefault();
                            const t = Object(i.c)(e.target);
                            window.setTimeout(() => {
                                this.show(t)
                            }, 0)
                        }
                    }), e(document).on("ajax:success", ".js-login-form", this.loginSuccess).on("ajax:error", ".js-login-form", this.loginError).on("submit", ".js-login-form", this.clearError).on("input", ".js-login-form-input", this.clearError).on("click", ".js-user-link", this.showOnClick).on("click", ".js-login-required--click", this.showToContinue).on("ajax:before", ".js-login-required--click", () => null != n.a.currentUser).on("ajax:error", this.onError).on("turbolinks:load", this.showOnLoad), e.subscribe("nav:popup:hidden", this.reset)
                }
            }
        }).call(this, s("5wds"))
    }, bSvc: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            var r = s("lv9K");

            class n {
                constructor() {
                    Object.defineProperty(this, "hasFocus", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: document.hasFocus()
                    }), Object(r.p)(this), e(window).on("blur focus", Object(r.f)(() => this.hasFocus = document.hasFocus()))
                }
            }

            (function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                i > 3 && a && Object.defineProperty(t, s, a)
            })([r.q], n.prototype, "hasFocus", void 0)
        }).call(this, s("5wds"))
    }, "bfq+": function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            })), s.d(t, "b", (function () {
                return l
            })), s.d(t, "c", (function () {
                return c
            })), s.d(t, "d", (function () {
                return u
            })), s.d(t, "e", (function () {
                return d
            })), s.d(t, "f", (function () {
                return p
            })), s.d(t, "g", (function () {
                return m
            })), s.d(t, "h", (function () {
                return h
            }));
            var r = s("0h6b"), n = s("lv9K"), i = s("d+cC"), a = s("f4vq");

            function o(t, s) {
                return e.post(Object(r.a)("chat.ack"), {history_since: s, since: t})
            }

            function l(t) {
                return e.get(Object(r.a)("chat.channels.show", {channel: t}))
            }

            function c(t, s) {
                return e.get(Object(r.a)("chat.channels.messages.index", Object.assign({
                    channel: t,
                    return_object: 1
                }, s))).then(Object(n.f)(e => (a.a.dataStore.userStore.updateWithJson(e.users), e.messages.map(e => i.a.fromJson(e)))))
            }

            function u(t, s) {
                return e.get(Object(r.a)("chat.updates"), {
                    history_since: s,
                    includes: ["presence", "silences"],
                    since: t
                })
            }

            function d(t, s) {
                return e.ajax({type: "PUT", url: Object(r.a)("chat.channels.mark-as-read", {channel: t, message: s})})
            }

            function p(t, s) {
                return e.post(Object(r.a)("chat.new"), {
                    is_action: s.isAction,
                    message: s.content,
                    target_id: t,
                    uuid: s.uuid
                })
            }

            function m(t, s) {
                return e.ajax(Object(r.a)("chat.channels.part", {channel: t, user: s}), {method: "DELETE"})
            }

            function h(t) {
                return e.post(Object(r.a)("chat.channels.messages.store", {channel: t.channelId}), {
                    is_action: t.isAction,
                    message: t.content,
                    target_id: t.channelId,
                    target_type: "channel",
                    uuid: t.uuid
                })
            }
        }).call(this, s("5wds"))
    }, bpgk: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
        }
    }, c1EF: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("tX/w");

        function i(e) {
            return r.createElement("span", {
                className: `${Object(n.a)("avatar", e.modifiers)} avatar--guest`,
                style: {backgroundImage: osu.urlPresence(e.user.avatar_url)}
            })
        }
    }, c9Hp: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("0h6b"), n = s("/G9H"), i = s("wsBb");

        function a(e) {
            const t = [{
                active: "index" === e.section,
                title: osu.trans("news.index.title.info"),
                url: Object(r.a)("news.index")
            }];
            return "show" === e.section && null != e.post && t.push({
                active: !0,
                title: e.post.title,
                url: Object(r.a)("news.show", {news: e.post.slug})
            }), n.createElement(i.a, {links: t, linksBreadcrumb: !0, theme: "news"})
        }
    }, cQQh: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("/G9H"), n = s("tX/w"), i = s("/HbY");

        class a extends r.PureComponent {
            get text() {
                return null == this.props.text ? null : "string" == typeof this.props.text ? {top: this.props.text} : this.props.text
            }

            render() {
                let e = Object(n.a)("btn-osu-big", this.props.modifiers, {disabled: this.props.disabled});
                return null != this.props.extraClasses && (e += ` ${this.props.extraClasses.join(" ")}`), osu.present(this.props.href) ? this.props.disabled ? r.createElement("span", Object.assign({className: e}, this.props.props), this.renderChildren()) : r.createElement("a", Object.assign({
                    className: e,
                    href: this.props.href
                }, this.props.props), this.renderChildren()) : r.createElement("button", Object.assign({
                    className: e,
                    disabled: this.props.disabled,
                    type: this.props.isSubmit ? "submit" : "button"
                }, this.props.props), this.renderChildren())
            }

            renderChildren() {
                const e = this.text;
                return r.createElement("span", {className: Object(n.a)("btn-osu-big__content", {center: null == e || null == this.props.icon})}, null != e && r.createElement("span", {className: "btn-osu-big__left"}, r.createElement("span", {className: "btn-osu-big__text-top"}, e.top), "bottom" in e && null != e.bottom && r.createElement("span", {className: "btn-osu-big__text-bottom"}, e.bottom)), null != this.props.icon && r.createElement("span", {className: "btn-osu-big__icon"}, r.createElement("span", {className: "fa fa-fw"}, this.props.isBusy ? r.createElement(i.a, {modifiers: "center-inline"}) : r.createElement("span", {className: this.props.icon}))))
            }
        }

        Object.defineProperty(a, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {disabled: !1, extraClasses: [], isBusy: !1, isSubmit: !1, props: {}}
        })
    }, cX0L: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        let r = 0;

        function n() {
            return ++r
        }
    }, ccO9: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return p
            }));
            var n, i = s("/G9H"), a = s("I8Ok"), o = s("tX/w"), l = s("vZz4"), c = s("aCRl"), u = function (e, t) {
                return function () {
                    return e.apply(t, arguments)
                }
            }, d = {}.hasOwnProperty;
            n = i.createElement;
            var p = function (t) {
                function s() {
                    return this.renderNav = u(this.renderNav, this), this.render = u(this.render, this), s.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) d.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.render = function () {
                    var t, s, i, u, d, p;
                    return t = Object(o.a)("build", this.props.modifiers), u = e.groupBy(this.props.build.changelog_entries, "category"), s = e(u).keys().sort().value(), Object(a.div)({className: t + " t-changelog-stream--" + e.kebabCase(this.props.build.update_stream.display_name)}, Object(a.div)({className: "build__version"}, this.renderNav({
                        version: "previous",
                        icon: "fas fa-chevron-left"
                    }), Object(a.a)({
                        className: "build__version-link",
                        href: Object(l.c)(this.props.build)
                    }, Object(a.span)({className: "build__stream"}, this.props.build.update_stream.display_name), " ", Object(a.span)({className: "u-changelog-stream--text"}, this.props.build.display_version)), this.renderNav({
                        version: "next",
                        icon: "fas fa-chevron-right"
                    })), null != (p = this.props.showDate) && p ? Object(a.div)({className: "build__date"}, r(this.props.build.created_at).format("LL")) : void 0, function () {
                        var e, t, r;
                        for (r = [], e = 0, t = s.length; e < t; e++) i = s[e], r.push(Object(a.div)({
                            key: i,
                            className: "build__changelog-entries-container"
                        }, Object(a.div)({className: "build__changelog-entries-category"}, i), Object(a.div)({className: "build__changelog-entries"}, function () {
                            var e, t, s, r, o;
                            for (o = [], e = 0, t = (s = u[i]).length; e < t; e++) d = s[e], o.push(Object(a.div)({
                                key: null != (r = d.id) ? r : d.created_at + "-" + d.title,
                                className: "build__changelog-entry"
                            }, n(c.a, {entry: d})));
                            return o
                        }())));
                        return r
                    }())
                }, s.prototype.renderNav = function (e) {
                    var t, s, r;
                    if (r = e.version, s = e.icon, null != this.props.build.versions) return null != (t = this.props.build.versions[r]) ? Object(a.a)({
                        className: "build__version-link",
                        href: Object(l.c)(t),
                        title: t.display_version
                    }, Object(a.i)({className: s})) : Object(a.span)({className: "build__version-link build__version-link--disabled"}, Object(a.i)({className: s}))
                }, s
            }(i.PureComponent)
        }).call(this, s("Hs9Z"), s("7EfK"))
    }, "ceX/": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return c
        }));
        var r = s("/G9H"), n = s("I8Ok"), i = s("+bOe"), a = function (e, t) {
            return function () {
                return e.apply(t, arguments)
            }
        }, o = {}.hasOwnProperty, l = [].indexOf || function (e) {
            for (var t = 0, s = this.length; t < s; t++) if (t in this && this[t] === e) return t;
            return -1
        }, c = function (e) {
            function t(e) {
                var s;
                this.toggleSelector = a(this.toggleSelector, this), this.renderOptions = a(this.renderOptions, this), this.renderOption = a(this.renderOption, this), this.render = a(this.render, this), this.optionSelected = a(this.optionSelected, this), this.hideSelector = a(this.hideSelector, this), this.componentDidUpdate = a(this.componentDidUpdate, this), this.componentDidMount = a(this.componentDidMount, this), t.__super__.constructor.call(this, e), this.bn = null != (s = this.props.bn) ? s : "select-options", this.hasBlackout = this.props.blackout || void 0 === this.props.blackout, this.ref = Object(r.createRef)(), this.state = {showingSelector: !1}
            }

            return function (e, t) {
                for (var s in t) o.call(t, s) && (e[s] = t[s]);

                function r() {
                    this.constructor = e
                }

                r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
            }(t, e), t.prototype.componentDidMount = function () {
                return document.addEventListener("click", this.hideSelector)
            }, t.prototype.componentDidUpdate = function (e, t) {
                if (this.hasBlackout && t.showingSelector !== this.state.showingSelector) return Object(i.c)(this.state.showingSelector, .5)
            }, t.prototype.componentWillUnmount = function () {
                return document.removeEventListener("click", this.hideSelector)
            }, t.prototype.hideSelector = function (e) {
                var t;
                if (0 === e.button && (t = this.ref.current, !(l.call(e.composedPath(), t) >= 0))) return this.setState({showingSelector: !1})
            }, t.prototype.optionSelected = function (e, t) {
                var s;
                if (0 === e.button) return e.preventDefault(), this.setState({showingSelector: !1}), "function" == typeof (s = this.props).onChange ? s.onChange(t) : void 0
            }, t.prototype.render = function () {
                var e, t;
                return e = "" + this.bn, this.state.showingSelector && (e += " " + this.bn + "--selecting"), Object(n.div)({
                    className: e,
                    ref: this.ref
                }, Object(n.div)({className: this.bn + "__select"}, this.renderOption({
                    children: [Object(n.div)({
                        className: "u-ellipsis-overflow",
                        key: "current"
                    }, null != (t = this.props.selected) ? t.text : void 0), Object(n.div)({
                        key: "decoration",
                        className: this.bn + "__decoration"
                    }, Object(n.i)({className: "fas fa-chevron-down"}))],
                    onClick: this.toggleSelector,
                    option: this.props.selected
                })), Object(n.div)({className: this.bn + "__selector"}, this.renderOptions()))
            }, t.prototype.renderOption = function (e) {
                var t, s, r, i, a, o;
                return t = e.children, r = e.onClick, i = e.option, o = null != (a = e.selected) && a, s = this.bn + "__option", o && (s += " " + this.bn + "__option--selected"), null != this.props.renderOption ? this.props.renderOption({
                    children: t,
                    cssClasses: s,
                    onClick: r,
                    option: i
                }) : Object(n.a)({className: s, href: "#", key: i.id, onClick: r}, t)
            }, t.prototype.renderOptions = function () {
                var e, t, s, r, i;
                for (i = [], e = 0, t = (r = this.props.options).length; e < t; e++) s = r[e], i.push(function (e) {
                    return function (t) {
                        var s;
                        return e.renderOption({
                            children: [Object(n.div)({
                                className: "u-ellipsis-overflow",
                                key: t.id
                            }, t.text)], onClick: function (s) {
                                return e.optionSelected(s, t)
                            }, option: t, selected: (null != (s = e.props.selected) ? s.id : void 0) === t.id
                        })
                    }
                }(this)(s));
                return i
            }, t.prototype.toggleSelector = function (e) {
                if (0 === e.button) return e.preventDefault(), this.setState({showingSelector: !this.state.showingSelector})
            }, t
        }(r.PureComponent)
    }, "cxU/": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("/G9H"), n = s("1BiD");

        function i(e) {
            const {groups: t = [], modifiers: s, short: i = !1, wrapper: a} = e;
            let o = !1;
            return r.createElement(r.Fragment, null, t.map(e => {
                const t = i && o ? `${a} u-hidden-narrow` : a;
                return o = !0, r.createElement("span", {
                    key: e.identifier,
                    className: t
                }, r.createElement(n.a, {group: e, modifiers: s}))
            }))
        }
    }, "d+cC": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return u
        }));
        var r = s("Hs9Z"), n = s("lv9K"), i = s("srn7"), a = s("7EfK"), o = s("f4vq"), l = s("vZz4"),
            c = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

        class u {
            constructor() {
                Object.defineProperty(this, "channelId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: -1
                }), Object.defineProperty(this, "content", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), Object.defineProperty(this, "errored", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isAction", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "messageId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: osu.uuid()
                }), Object.defineProperty(this, "persisted", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "senderId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: -1
                }), Object.defineProperty(this, "timestamp", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: a().toISOString()
                }), Object.defineProperty(this, "uuid", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: this.messageId
                }), Object.defineProperty(this, "contentHtml", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object(n.p)(this)
            }

            get parsedContent() {
                var e;
                return null !== (e = this.contentHtml) && void 0 !== e ? e : Object(l.g)(Object(r.escape)(this.content), !0)
            }

            get sender() {
                var e;
                return null !== (e = o.a.dataStore.userStore.get(this.senderId)) && void 0 !== e ? e : new i.a(-1)
            }

            static fromJson(e) {
                var t;
                const s = new u;
                return s.channelId = e.channel_id, s.content = e.content, s.contentHtml = e.content_html, s.isAction = e.is_action, s.messageId = e.message_id, s.persisted = !0, s.senderId = e.sender_id, s.timestamp = e.timestamp, s.uuid = null !== (t = e.uuid) && void 0 !== t ? t : s.uuid, s
            }

            persist(e) {
                this.persisted || (this.messageId = e.message_id, this.timestamp = e.timestamp, this.persisted = !0)
            }
        }

        c([n.q], u.prototype, "channelId", void 0), c([n.q], u.prototype, "content", void 0), c([n.q], u.prototype, "errored", void 0), c([n.q], u.prototype, "isAction", void 0), c([n.q], u.prototype, "messageId", void 0), c([n.q], u.prototype, "persisted", void 0), c([n.q], u.prototype, "senderId", void 0), c([n.q], u.prototype, "timestamp", void 0), c([n.q], u.prototype, "uuid", void 0), c([n.q], u.prototype, "contentHtml", void 0), c([n.h], u.prototype, "parsedContent", null), c([n.h], u.prototype, "sender", null), c([n.f], u.prototype, "persist", null)
    }, dC65: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return r
            }));

            class r {
                constructor() {
                    Object.defineProperty(this, "createTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            e(t).qtip({
                                content: {text: s},
                                hide: {
                                    delay: 200, effect() {
                                        e(this).fadeTo(110, 0)
                                    }, fixed: !0
                                },
                                position: {at: "top center", my: "bottom center", viewport: e(window)},
                                show: {
                                    delay: 200, effect() {
                                        e(this).fadeTo(110, 1)
                                    }, ready: !0
                                },
                                style: {classes: "tooltip-default tooltip-default--interactable"}
                            })
                        }
                    }), Object.defineProperty(this, "showTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            var t;
                            if (!(e.currentTarget instanceof HTMLAnchorElement)) return;
                            const s = e.currentTarget, r = s.getAttribute("href");
                            if (null == r) return;
                            const n = null === (t = document.querySelector(r)) || void 0 === t ? void 0 : t.firstElementChild;
                            if (!(n instanceof HTMLParagraphElement)) return;
                            const i = document.createElement("div");
                            i.insertAdjacentHTML("afterbegin", n.innerHTML), i.querySelectorAll("*").forEach(e => {
                                "doc-backlink" === e.getAttribute("role") ? e.remove() : e.removeAttribute("class")
                            }), this.createTooltip(s, i)
                        }
                    }), e(document).on("mouseover", ".js-reference-link", this.showTooltip)
                }
            }
        }).call(this, s("5wds"))
    }, dK4O: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("0h6b"), n = s("/G9H"), i = s("tX/w");

        class a extends n.PureComponent {
            get artist() {
                var e;
                return "artist" in this.props ? null !== (e = this.props.track.artist) && void 0 !== e ? e : this.props.artist : this.props.track.artist
            }

            render() {
                let e = Object(i.a)("artist-track", {original: this.props.track.exclusive}, this.props.modifiers);
                return e += " js-audio--player", n.createElement("div", {
                    className: e,
                    "data-audio-url": this.props.track.preview
                }, n.createElement("div", {
                    className: "artist-track__col artist-track__col--preview",
                    style: {backgroundImage: osu.urlPresence(this.props.track.cover_url)}
                }, n.createElement("button", {className: "artist-track__button artist-track__button--play js-audio--play"}, n.createElement("span", {className: "fa-fw play-button"}))), n.createElement("div", {className: "artist-track__col artist-track__col--names"}, n.createElement("div", {className: "artist-track__title u-ellipsis-overflow"}, this.props.track.title, osu.present(this.props.track.version) && n.createElement(n.Fragment, null, " ", n.createElement("span", {className: "artist-track__version"}, this.props.track.version))), n.createElement("div", {className: "artist-track__info"}, n.createElement("a", {href: Object(r.a)("artists.show", {artist: this.artist.id})}, this.artist.name)), this.props.showAlbum && null != this.props.track.album && n.createElement("div", {className: "artist-track__info"}, n.createElement("a", {href: `${Object(r.a)("artists.show", {artist: this.artist.id})}#album-${this.props.track.album_id}`}, this.props.track.album.title))), n.createElement("div", {className: "artist-track__col artist-track__col--badges"}, this.props.track.exclusive && n.createElement("span", {
                    className: "pill-badge pill-badge--pink pill-badge--with-shadow",
                    title: osu.trans("artist.songs.original")
                }, osu.trans("artist.songs.original_badge")), this.props.track.is_new && n.createElement("span", {className: "pill-badge pill-badge--yellow pill-badge--with-shadow"}, osu.trans("common.badges.new"))), n.createElement("div", {className: "artist-track__col artist-track__col--details"}, n.createElement("div", {className: "u-ellipsis-overflow artist-track__detail artist-track__detail--genre"}, this.props.track.genre), n.createElement("div", {className: "artist-track__detail artist-track__detail--bpm"}, osu.formatNumber(this.props.track.bpm), "bpm"), n.createElement("div", {className: "artist-track__detail artist-track__detail--length"}, this.props.track.length)), n.createElement("div", {className: "artist-track__col artist-track__col--buttons"}, n.createElement("a", {
                    className: "artist-track__button",
                    href: this.props.track.osz,
                    title: osu.trans("artist.beatmaps.download")
                }, n.createElement("span", {className: "fas fa-fw fa-download"}))))
            }
        }

        Object.defineProperty(a, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {showAlbum: !1}
        })
    }, dMN7: function (e, t, s) {
        "use strict";

        function r(e) {
            return null != e.accuracy
        }

        s.d(t, "a", (function () {
            return r
        }))
    }, eIGF: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return r
            }));

            class r {
                constructor() {
                    Object.defineProperty(this, "switchEdit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = e(t.target).parents(".js-forum-poll--container");
                            if ("1" === s.attr("data-edit")) {
                                s.attr("data-edit", "0");
                                const r = e(t.target).closest("form")[0];
                                null != r && r.reset()
                            } else s.attr("data-edit", "1")
                        }
                    }), Object.defineProperty(this, "switchPage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            const s = t.currentTarget;
                            if (!(s instanceof HTMLElement)) return;
                            const r = s.dataset.targetPage;
                            "string" == typeof r && e(t.target).parents(".js-forum-poll--container").attr("data-page", r)
                        }
                    }), e(document).on("click", ".js-forum-poll--switch-page", this.switchPage).on("click", ".js-forum-poll--switch-edit", this.switchEdit)
                }
            }
        }).call(this, s("5wds"))
    }, elBb: function (e, t, s) {
        "use strict";

        function r(e) {
            return null != e ? JSON.parse(JSON.stringify(e)) : e
        }

        function n(e) {
            const t = i(e);
            if (null == t) throw new Error(`script element ${e} is missing or contains nullish value.`);
            return t
        }

        function i(e, t = !1) {
            var s;
            const r = null === (s = window.newBody) || void 0 === s ? void 0 : s.querySelector(`#${e}`);
            if (!(r instanceof HTMLScriptElement)) return;
            const n = JSON.parse(r.text);
            return t && r.remove(), n
        }

        function a(e, t) {
            const s = JSON.stringify(t), r = document.getElementById(e);
            let n;
            if (null == r) (n = document.createElement("script")).id = e, n.type = "application/json", document.body.appendChild(n); else {
                if (!(r instanceof HTMLScriptElement)) throw new Error(`existing ${e} is not a script element.`);
                n = r
            }
            n.text = s
        }

        s.d(t, "a", (function () {
            return r
        })), s.d(t, "b", (function () {
            return n
        })), s.d(t, "c", (function () {
            return i
        })), s.d(t, "d", (function () {
            return a
        }))
    }, f4vq: function (e, t, s) {
        "use strict";
        const r = new (s("BSzr").a);
        window.osuCore = r, t.a = r
    }, gcbN: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("/G9H"), n = s("tX/w"), i = s("rLgU"), a = s("ss8h");
        const o = e => {
            const t = r.useContext(a.a), s = r.useCallback(s => {
                s.preventDefault(), Object(i.h)(t, e.format)
            }, [t, e.format]);
            return r.createElement("button", {
                className: Object(n.a)("beatmap-discussion-editor-toolbar__button", [Object(i.d)(t, e.format) ? "active" : ""]),
                onMouseDown: s
            }, r.createElement("i", {className: `fas fa-${e.format}`}))
        }
    }, "h/Ip": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = Object(r.createContext)({excludes: [], isWidget: !1})
    }, hoYT: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));
        const r = ["osu", "taiko", "fruits", "mania"]
    }, i41Q: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return v
            }));
            var n, i, a, o = s("0h6b"), l = s("f4vq"), c = s("/G9H"), u = s("I8Ok"), d = s("tX/w"), p = s("R9Sp"),
                m = s("/HbY"), h = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, b = {}.hasOwnProperty, f = [].indexOf || function (e) {
                    for (var t = 0, s = this.length; t < s; t++) if (t in this && this[t] === e) return t;
                    return -1
                };
            i = c.createElement, a = l.a.dataStore.uiState, n = "comment-show-more";
            var v = function (t) {
                function s(e) {
                    this.load = h(this.load, this), this.render = h(this.render, this), this.componentWillUnmount = h(this.componentWillUnmount, this), s.__super__.constructor.call(this, e), this.state = {loading: !1}
                }

                return function (e, t) {
                    for (var s in t) b.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.defaultProps = {modifiers: []}, s.prototype.componentWillUnmount = function () {
                    var e;
                    return null != (e = this.xhr) ? e.abort() : void 0
                }, s.prototype.render = function () {
                    var e, t, s, r, o, l, c;
                    return this.props.comments.length >= this.props.total ? null : null == (s = a.comments.hasMoreComments[null != (r = null != (o = this.props.parent) ? o.id : void 0) ? r : null]) || s ? (e = Object(d.a)(n, this.props.modifiers), f.call(this.props.modifiers, "top") >= 0 ? (c = this.props.total - this.props.comments.length, t = ["comments"], f.call(this.props.modifiers, "changelog") >= 0 ? t.push("t-greyviolet-darker") : t.push("t-ddd"), i(p.a, {
                        loading: this.state.loading,
                        hasMore: !0,
                        callback: this.load,
                        modifiers: t,
                        remaining: c
                    })) : Object(u.div)({className: e}, this.state.loading ? i(m.a) : Object(u.button)({
                        className: n + "__link",
                        onClick: this.load
                    }, null != (l = this.props.label) ? l : osu.trans("common.buttons.show_more")))) : null
                }, s.prototype.load = function () {
                    var t, s, n, i, l, c, u, d, p;
                    return this.setState({loading: !0}), s = {
                        commentable_type: null != (n = null != (i = this.props.parent) ? i.commentable_type : void 0) ? n : this.props.commentableType,
                        commentable_id: null != (l = null != (c = this.props.parent) ? c.commentable_id : void 0) ? l : this.props.commentableId,
                        parent_id: null != (u = null != (d = this.props.parent) ? d.id : void 0) ? u : 0,
                        sort: a.comments.currentSort
                    }, null != (t = e.last(this.props.comments)) && (s.cursor = {
                        id: t.id,
                        created_at: t.createdAt,
                        votes_count: t.votesCount
                    }), this.xhr = r.ajax(Object(o.a)("comments.index"), {
                        data: s,
                        dataType: "json"
                    }).done((function (e) {
                        return r.publish("comments:added", e)
                    })).always((p = this, function () {
                        return p.setState({loading: !1})
                    }))
                }, s
            }(c.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, iCid: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return r
            }));

            class r {
                constructor() {
                    Object.defineProperty(this, "observer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new MutationObserver(r.handleMutations)
                    }), this.observer.observe(document, {
                        attributeFilter: ["datetime"],
                        attributeOldValue: !0,
                        childList: !0,
                        subtree: !0
                    })
                }
            }

            Object.defineProperty(r, "className", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: "js-timeago"
            }), Object.defineProperty(r, "searchQuery", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: `.${r.className}`
            }), Object.defineProperty(r, "handleMutation", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: t => {
                    switch (t.type) {
                        case"childList":
                            e(t.addedNodes).find(r.searchQuery).addBack(r.searchQuery).timeago();
                            break;
                        case"attributes":
                            t.target instanceof HTMLTimeElement && t.target.dateTime !== t.oldValue && t.target.classList.contains(r.className) && e(t.target).timeago("updateFromDOM")
                    }
                }
            }), Object.defineProperty(r, "handleMutations", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: t => {
                    null != e.fn.timeago && t.forEach(r.handleMutation)
                }
            })
        }).call(this, s("5wds"))
    }, "iH//": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return l
        })), s.d(t, "b", (function () {
            return c
        })), s.d(t, "c", (function () {
            return u
        }));
        var r = Object.freeze({
            colour: "hsl(var(--hsl-l1))",
            has_listing: !1,
            has_playmodes: !1,
            id: -1,
            identifier: "guest",
            is_probationary: !1,
            name: "Difficulty Guest",
            short_name: "GUEST"
        }), n = s("VxeA"), i = s("is6n"), a = s("vZz4"), o = function (e, t) {
            var s = {};
            for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (s[r] = e[r]);
            if (null != e && "function" == typeof Object.getOwnPropertySymbols) {
                var n = 0;
                for (r = Object.getOwnPropertySymbols(e); n < r.length; n++) t.indexOf(r[n]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[n]) && (s[r[n]] = e[r[n]])
            }
            return s
        };

        function l({beatmapset: e, currentBeatmap: t, discussion: s, user: i}) {
            var a;
            return null == i ? null : i.id === e.user_id ? n.a : null != t && s.beatmap_id === t.id && i.id === t.user_id ? r : null === (a = i.groups) || void 0 === a ? void 0 : a[0]
        }

        function c(e) {
            return e.replace(a.j, (e, t) => {
                const s = u(t), {children: r} = s, n = o(s, ["children"]), i = "string" == typeof r ? r : t;
                return Object(a.f)(t, i, {props: n, unescape: !0})
            })
        }

        function u(e) {
            const t = BeatmapDiscussionHelper.urlParse(Object(i.a)().href),
                s = {children: e, rel: "nofollow noreferrer", target: "_blank"};
            let r;
            try {
                r = new URL(e)
            } catch (n) {
            }
            if (null != r && r.host === Object(i.a)().host) {
                const e = BeatmapDiscussionHelper.urlParse(r.href, null, {forceDiscussionId: !0});
                if (null != (null == e ? void 0 : e.discussionId) && null != e.beatmapsetId) {
                    const r = [e.discussionId, e.postId].filter(Number.isFinite).join("/");
                    (null == t ? void 0 : t.beatmapsetId) === e.beatmapsetId ? (s.children = `#${r}`, s.className = "js-beatmap-discussion--jump", s.target = void 0) : s.children = `${e.beatmapsetId}#${r}`
                }
            }
            return s
        }
    }, is6n: function (e, t, s) {
        "use strict";

        function r() {
            var e;
            return null !== (e = window.newUrl) && void 0 !== e ? e : document.location
        }

        function n() {
            const e = r();
            return e instanceof URL ? e.searchParams : new URLSearchParams(e.search)
        }

        function i() {
            const e = r();
            return `${e.pathname}${e.search}`
        }

        s.d(t, "a", (function () {
            return r
        })), s.d(t, "b", (function () {
            return n
        })), s.d(t, "c", (function () {
            return i
        }))
    }, jUJ3: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        let r = null;

        function n(e) {
            var t;
            const s = function () {
                if (null == r && null == (r = document.querySelector(".js-estimate-min-lines"))) throw new Error("js-estimate-min-lines placeholder element is missing!");
                return r
            }();
            s.innerHTML = e;
            const n = s.firstChild;
            let i = s;
            n instanceof HTMLElement && (i = n);
            const a = parseFloat(null !== (t = window.getComputedStyle(i).getPropertyValue("line-height")) && void 0 !== t ? t : "0"),
                o = s.scrollHeight;
            return {count: Math.ceil(o / a), height: o, lineHeight: a}
        }
    }, "k5H+": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("Hs9Z"), n = s("/G9H"), i = s("tX/w");

        class a extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "bodyRef", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: n.createRef()
                }), Object.defineProperty(this, "sizeSelectRef", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: n.createRef()
                }), Object.defineProperty(this, "cancel", {
                    enumerable: !0, configurable: !0, writable: !0, value: e => {
                        var t;
                        ((null === (t = this.bodyRef.current) || void 0 === t ? void 0 : t.value) === this.props.rawValue || confirm(osu.trans("common.confirmation_unsaved"))) && (null != this.bodyRef.current && (this.bodyRef.current.value = this.props.rawValue), this.sendOnChange({
                            event: e,
                            type: "cancel"
                        }))
                    }
                }), Object.defineProperty(this, "onKeyDown", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        27 === e.keyCode && this.cancel()
                    }
                }), Object.defineProperty(this, "save", {
                    enumerable: !0, configurable: !0, writable: !0, value: e => {
                        this.sendOnChange({event: e, type: "save"})
                    }
                })
            }

            componentDidMount() {
                null != this.sizeSelectRef.current && (this.sizeSelectRef.current.value = ""), null != this.bodyRef.current && (this.bodyRef.current.selectionEnd = 0, this.bodyRef.current.focus())
            }

            render() {
                let e = Object(i.a)("bbcode-editor", this.props.modifiers);
                return e += " js-bbcode-preview--form", n.createElement("form", {
                    className: e,
                    "data-state": "write"
                }, n.createElement("div", {className: "bbcode-editor__content"}, n.createElement("textarea", {
                    ref: this.bodyRef,
                    className: "bbcode-editor__body js-bbcode-preview--body",
                    defaultValue: this.props.rawValue,
                    disabled: this.props.disabled,
                    name: "body",
                    onKeyDown: this.onKeyDown,
                    placeholder: this.props.placeholder
                }), n.createElement("div", {className: "bbcode-editor__preview"}, n.createElement("div", {className: "forum-post-content js-bbcode-preview--preview"})), n.createElement("div", {className: "bbcode-editor__buttons-bar"}, n.createElement("div", {className: "bbcode-editor__buttons bbcode-editor__buttons--toolbar"}, this.renderToolbar()), n.createElement("div", {className: "bbcode-editor__buttons bbcode-editor__buttons--actions"}, n.createElement("div", {className: "bbcode-editor__button bbcode-editor__button--cancel"}, this.actionButton(this.cancel, osu.trans("common.buttons.cancel"))), n.createElement("div", {className: "bbcode-editor__button bbcode-editor__button--hide-on-write"}, this.renderPreviewHideButton()), n.createElement("div", {className: "bbcode-editor__button bbcode-editor__button--hide-on-preview"}, this.renderPreviewShowButton()), n.createElement("div", {className: "bbcode-editor__button"}, this.actionButton(this.save, osu.trans("common.buttons.save"), "forum-primary"))))))
            }

            actionButton(e, t, s = "forum-secondary") {
                return n.createElement("button", {
                    className: Object(i.a)("btn-osu-big", s),
                    disabled: this.props.disabled,
                    onClick: e,
                    type: "button"
                }, t)
            }

            renderPreviewHideButton() {
                return n.createElement("button", {
                    className: "js-bbcode-preview--hide btn-osu-big btn-osu-big--forum-secondary",
                    disabled: this.props.disabled,
                    type: "button"
                }, osu.trans("forum.topic.create.preview_hide"))
            }

            renderPreviewShowButton() {
                return n.createElement("button", {
                    className: "js-bbcode-preview--show btn-osu-big btn-osu-big--forum-secondary",
                    disabled: this.props.disabled,
                    type: "button"
                }, osu.trans("forum.topic.create.preview"))
            }

            renderToolbar() {
                return n.createElement("div", {className: "post-box-toolbar"}, this.toolbarButton("bold", n.createElement("i", {className: "fas fa-bold"})), this.toolbarButton("italic", n.createElement("i", {className: "fas fa-italic"})), this.toolbarButton("strikethrough", n.createElement("i", {className: "fas fa-strikethrough"})), this.toolbarButton("heading", n.createElement("i", {className: "fas fa-heading"})), this.toolbarButton("link", n.createElement("i", {className: "fas fa-link"})), this.toolbarButton("spoilerbox", n.createElement("i", {className: "fas fa-barcode"})), this.toolbarButton("list-numbered", n.createElement("i", {className: "fas fa-list-ol"})), this.toolbarButton("list", n.createElement("i", {className: "fas fa-list"})), this.toolbarButton("image", n.createElement("i", {className: "fas fa-image"})), this.toolbarSizeSelect())
            }

            sendOnChange({event: e, type: t}) {
                var s, r;
                this.props.onChange({
                    event: e,
                    hasChanged: (null === (s = this.bodyRef.current) || void 0 === s ? void 0 : s.value) !== this.props.rawValue,
                    type: t,
                    value: null === (r = this.bodyRef.current) || void 0 === r ? void 0 : r.value
                })
            }

            toolbarButton(e, t) {
                return n.createElement("button", {
                    className: `btn-circle btn-circle--bbcode js-bbcode-btn--${e}`,
                    disabled: this.props.disabled,
                    title: osu.trans(`bbcode.${Object(r.snakeCase)(e)}`),
                    type: "button"
                }, n.createElement("span", {className: "btn-circle__content"}, t))
            }

            toolbarSizeSelect() {
                return n.createElement("label", {
                    className: "bbcode-size-select",
                    title: osu.trans("bbcode.size._")
                }, n.createElement("span", {className: "bbcode-size-select__label"}, osu.trans("bbcode.size._")), n.createElement("i", {className: "fas fa-chevron-down"}), n.createElement("select", {
                    ref: this.sizeSelectRef,
                    className: "bbcode-size-select__select js-bbcode-btn--size",
                    disabled: this.props.disabled
                }, n.createElement("option", {value: "50"}, osu.trans("bbcode.size.tiny")), n.createElement("option", {value: "85"}, osu.trans("bbcode.size.small")), n.createElement("option", {value: "100"}, osu.trans("bbcode.size.normal")), n.createElement("option", {value: "150"}, osu.trans("bbcode.size.large"))))
            }
        }

        Object.defineProperty(a, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {disabled: !1}
        })
    }, kD1C: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return o
            }));
            var r = s("Hs9Z"), n = s("f4vq"), i = s("/DQ7"), a = s("phBA");

            class o {
                constructor() {
                    Object.defineProperty(this, "debouncedOnScroll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: Object(r.debounce)(() => this.onScroll(), 20)
                    }), Object.defineProperty(this, "onScroll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.pin(), this.stickOrUnstick()
                        }
                    }), Object.defineProperty(this, "stickOrUnstick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = this.shouldStick;
                            null != e && this.setVisible(e)
                        }
                    }), e(window).on("scroll", this.onScroll), e(document).on("turbolinks:load", this.debouncedOnScroll), e.subscribe("osu:page:change", this.debouncedOnScroll), e(window).on("resize", this.stickOrUnstick)
                }

                get breadcrumbsElement() {
                    var e;
                    return null === (e = window.newBody) || void 0 === e ? void 0 : e.querySelector(".js-sticky-header-breadcrumbs")
                }

                get contentElement() {
                    var e;
                    return null === (e = window.newBody) || void 0 === e ? void 0 : e.querySelector(".js-sticky-header-content")
                }

                get headerHeight() {
                    const e = window._styles.header;
                    return n.a.windowSize.isMobile ? e.heightMobile : e.heightSticky
                }

                get header() {
                    return document.querySelector(".js-pinned-header")
                }

                get marker() {
                    return document.querySelector(".js-sticky-header")
                }

                get pinnedSticky() {
                    return document.querySelector(".js-pinned-header-sticky")
                }

                get scrollOffsetValue() {
                    const e = this.pinnedSticky, t = null == e ? 0 : e.getBoundingClientRect().height;
                    return this.headerHeight + t
                }

                get shouldStick() {
                    const e = this.marker, t = this.pinnedSticky;
                    if (null != e && null != t) return e.getBoundingClientRect().top < this.headerHeight + t.getBoundingClientRect().height
                }

                scrollOffset(e) {
                    return Math.max(0, e - this.scrollOffsetValue)
                }

                pin() {
                    null != this.header && (this.shouldPin() ? document.body.classList.add("js-header-is-pinned") : document.body.classList.remove("js-header-is-pinned"))
                }

                setVisible(t) {
                    Object(i.c)(Object(a.e)(this.pinnedSticky), t), e(document).trigger("sticky-header:sticking", [t])
                }

                shouldPin(e) {
                    return (null != e ? e : window.pageYOffset) > 30 || this.shouldStick
                }
            }
        }).call(this, s("5wds"))
    }, kXXC: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
            constructor(e) {
                Object.defineProperty(this, "userId", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }
    }, kiUL: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return d
            }));
            var r = s("WLnA"), n = s("0h6b"), i = s("Hs9Z"), a = s("lv9K"), o = s("y2EG"), l = s("nxXY"), c = s("uW+8"),
                u = function (e, t, s, r) {
                    var n, i = arguments.length,
                        a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                };

            class d {
                constructor() {
                    Object.defineProperty(this, "debouncedDeleteByIds", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: Object(i.debounce)(this.deleteByIds, 500)
                    }), Object.defineProperty(this, "debouncedSendQueuedMarkedAsRead", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: Object(i.debounce)(this.sendQueuedMarkedAsRead, 500)
                    }), Object.defineProperty(this, "deleteByIdsQueue", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "queuedMarkedAsRead", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "queuedMarkedAsReadIdentities", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object(a.p)(this)
                }

                delete(t) {
                    if (t.isDeleting = !0, t instanceof o.a) return this.deleteByIdsQueue.set(t.id, t), void this.debouncedDeleteByIds();
                    e.ajax({
                        data: {identities: [Object(l.d)(t.identity)]},
                        dataType: "json",
                        method: "DELETE",
                        url: Object(n.a)("notifications.index")
                    }).then(Object(a.f)(() => {
                        Object(r.a)(new c.a([t.identity], 0))
                    })).catch(osu.ajaxError).always(Object(a.f)(() => t.isDeleting = !1))
                }

                loadMore(t, s, i) {
                    const a = Object(l.d)(t);
                    delete a.id;
                    const o = Object(n.a)("notifications.index", a),
                        u = {data: {cursor: i, unread: s.isWidget}, dataType: "json", url: o};
                    return e.ajax(u).then(e => {
                        Object(r.a)(new c.b(e, s))
                    })
                }

                queueMarkAsRead(e) {
                    e.isMarkingAsRead = !0;
                    const t = e.identity;
                    "stack" !== Object(l.b)(t) ? (e instanceof o.a && e.canMarkRead ? this.queuedMarkedAsRead.set(e.id, e) : this.queuedMarkedAsReadIdentities.set(Object(l.e)(t), e), this.debouncedSendQueuedMarkedAsRead()) : this.sendMarkAsReadRequest({identities: [Object(l.d)(e.identity)]}).then(Object(a.f)(() => {
                        Object(r.a)(new c.d([t], 0))
                    })).always(Object(a.f)(() => e.isMarkingAsRead = !1))
                }

                deleteByIds() {
                    if (0 === this.deleteByIdsQueue.size) return;
                    const t = [...this.deleteByIdsQueue.values()], s = t.map(e => e.identity);
                    this.deleteByIdsQueue.clear(), e.ajax({
                        data: {notifications: s.map(l.d)},
                        dataType: "json",
                        method: "DELETE",
                        url: Object(n.a)("notifications.index")
                    }).then(Object(a.f)(() => {
                        Object(r.a)(new c.a(s, 0))
                    })).always(Object(a.f)(() => t.forEach(e => e.isDeleting = !1)))
                }

                sendMarkAsReadRequest(t) {
                    return e.ajax({
                        data: t,
                        dataType: "json",
                        method: "POST",
                        url: Object(n.a)("notifications.mark-read")
                    }).catch(osu.ajaxError)
                }

                sendQueuedMarkedAsRead() {
                    if (this.queuedMarkedAsRead.size > 0) {
                        const e = [...this.queuedMarkedAsRead.values()], t = e.map(e => e.identity);
                        this.queuedMarkedAsRead.clear(), this.sendMarkAsReadRequest({notifications: t.map(l.d)}).then(Object(a.f)(() => {
                            Object(r.a)(new c.d(t, 0))
                        })).always(Object(a.f)(() => e.forEach(e => e.isMarkingAsRead = !1)))
                    }
                    if (this.queuedMarkedAsReadIdentities.size > 0) {
                        const e = [...this.queuedMarkedAsReadIdentities.values()], t = e.map(e => e.identity);
                        this.queuedMarkedAsReadIdentities.clear(), this.sendMarkAsReadRequest({identities: t.map(l.d)}).then(Object(a.f)(() => {
                            Object(r.a)(new c.d(t, 0))
                        })).always(Object(a.f)(() => e.forEach(e => e.isMarkingAsRead = !1)))
                    }
                }
            }

            u([a.f], d.prototype, "delete", null), u([a.f], d.prototype, "loadMore", null), u([a.f], d.prototype, "queueMarkAsRead", null)
        }).call(this, s("5wds"))
    }, mArn: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            const r = "//platform.enchant.com";

            class n {
                constructor(t) {
                    Object.defineProperty(this, "turbolinksReload", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "load", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null != document.querySelector(".enchant-help-center") && (window.enchant = [], this.turbolinksReload.load(r))
                        }
                    }), Object.defineProperty(this, "showMessageWindow", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            e.preventDefault(), null != window.enchant && null != window.enchant.messenger && "function" == typeof window.enchant.messenger.open && window.enchant.messenger.open()
                        }
                    }), Object.defineProperty(this, "unload", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.turbolinksReload.forget(r), e('#enchant-messenger-main, #enchant-messenger-launcher, iframe[src^="https://enchantwidgets-"]').remove(), document.querySelectorAll("style").forEach(t => {
                                const s = t.textContent;
                                null != s && /#enchant-/.exec(s) && e(t).remove()
                            })
                        }
                    }), e(document).on("turbolinks:load", this.load), e(document).on("turbolinks:before-cache", this.unload), e(document).on("click", ".js-enchant--show", this.showMessageWindow)
                }
            }
        }).call(this, s("5wds"))
    }, mjdM: function (e, t, s) {
        "use strict";
        var r = s("UBw1"), n = s("KUml"), i = s("h/Ip"), a = s("8Xmz"), o = s("9zVE"), l = s("/G9H"), c = s("tX/w"),
            u = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };
        let d = class extends l.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "handleContainerClick", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        osu.isClickable(e.target) || null != this.props.markRead && this.props.markRead()
                    }
                })
            }

            get canMarkAsRead() {
                var e;
                return null !== (e = this.props.canMarkAsRead) && void 0 !== e ? e : this.props.item.canMarkRead
            }

            render() {
                return l.createElement("div", {
                    className: this.blockClass(),
                    onClick: this.handleContainerClick
                }, this.renderCover(), l.createElement("div", {className: "notification-popup-item__main"}, l.createElement("div", {className: "notification-popup-item__content"}, this.renderCategory(), this.renderMessage(), this.renderTime(), this.renderExpandButton()), this.renderMarkAsReadButton(), this.renderDeleteButton()), this.renderUnreadStripe())
            }

            blockClass() {
                const e = [...this.props.modifiers, this.props.item.category];
                return this.props.item.isRead && !this.props.canMarkAsRead && e.push("read"), `clickable-row ${Object(c.a)("notification-popup-item", e)}`
            }

            renderCategory() {
                if (!this.props.withCategory) return null;
                const e = osu.trans(`notifications.item.${this.props.item.displayType}.${this.props.item.category}._`);
                return "" === e ? null : l.createElement("div", {className: "notification-popup-item__row notification-popup-item__row--category"}, e)
            }

            renderCover() {
                const e = this.props.withCoverImage ? this.props.item.details.coverUrl : null;
                return l.createElement("div", {
                    className: "notification-popup-item__cover",
                    style: {backgroundImage: osu.urlPresence(e)}
                }, l.createElement("div", {className: "notification-popup-item__cover-overlay"}, this.renderCoverIcons()))
            }

            renderCoverIcons() {
                return null == this.props.icons ? null : this.props.icons.map(e => l.createElement("div", {
                    key: e,
                    className: "notification-popup-item__cover-icon"
                }, l.createElement("span", {className: e})))
            }

            renderDeleteButton() {
                var e;
                return this.context.isWidget ? null : l.createElement(a.a, {
                    isDeleting: null !== (e = this.props.isDeleting) && void 0 !== e ? e : this.props.item.isDeleting,
                    modifiers: ["fancy"],
                    onDelete: this.props.delete
                })
            }

            renderExpandButton() {
                return null == this.props.expandButton ? null : l.createElement("div", {className: "notification-popup-item__row notification-popup-item__row--expand"}, this.props.expandButton)
            }

            renderMarkAsReadButton() {
                var e;
                return this.canMarkAsRead ? l.createElement(o.a, {
                    isMarkingAsRead: null !== (e = this.props.isMarkingAsRead) && void 0 !== e ? e : this.props.item.isMarkingAsRead,
                    modifiers: ["fancy"],
                    onMarkAsRead: this.props.markRead
                }) : null
            }

            renderMessage() {
                return l.createElement("a", {
                    className: "notification-popup-item__row notification-popup-item__row--message clickable-row-link",
                    href: this.props.url,
                    onClick: this.props.markRead
                }, this.props.message)
            }

            renderTime() {
                if (null != this.props.item.createdAtJson) return l.createElement("div", {className: "notification-popup-item__row notification-popup-item__row--time"}, l.createElement(r.a, {
                    dateTime: this.props.item.createdAtJson,
                    relative: !0
                }))
            }

            renderUnreadStripe() {
                return this.context.isWidget || !this.canMarkAsRead ? null : l.createElement("span", {className: "notification-popup-item__unread-stripe"})
            }
        };
        Object.defineProperty(d, "contextType", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: i.a
        }), d = u([n.b], d), t.a = d
    }, mkgZ: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));

        class r {
        }
    }, msVt: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("lv9K"), n = s("JsCm"), i = s("5e07");
            const a = () => {
                const e = document.createElement("div");
                return e.className = "audio-player audio-player--main", e.innerHTML = `\n    <button\n      type="button"\n      class="audio-player__button audio-player__button--prev js-audio--nav"\n      data-audio-nav="prev"\n    ><span class="fas fa-fw fa-step-backward"></span></button>\n\n    <button\n      type="button"\n      class="audio-player__button audio-player__button--play js-audio--main-play"\n    ><span class="fa-fw play-button"></span></button>\n\n    <button\n      type="button"\n      class="audio-player__button audio-player__button--next js-audio--nav"\n      data-audio-nav="next"\n    ><span class="fas fa-fw fa-step-forward"></span></button>\n\n    <div class="audio-player__bar audio-player__bar--progress js-audio--seek">\n      <div class="audio-player__bar-current"></div>\n    </div>\n\n    <div class="audio-player__timestamps">\n      <div class="audio-player__timestamp audio-player__timestamp--current"></div>\n      <div class="audio-player__timestamp-separator">/</div>\n      <div class="audio-player__timestamp audio-player__timestamp--total"></div>\n    </div>\n\n    <div class="audio-player__volume-control">\n      <button type="button" class="audio-player__volume-button js-audio--toggle-mute"></button>\n      <div class="audio-player__bar audio-player__bar--volume js-audio--volume">\n        <div class="audio-player__bar-current"></div>\n      </div>\n    </div>\n\n    <div class="audio-player__autoplay-control">\n      <button type="button" class="audio-player__autoplay-button js-audio--toggle-autoplay" title="${osu.trans("layout.audio.autoplay")}"></button>\n    </div>\n  `, e
            }, o = () => {
                const e = document.createElement("div");
                return e.className = "audio-player js-audio--player", e.innerHTML = '\n    <button\n    type="button"\n    class="audio-player__button audio-player__button--play js-audio--play"\n    ><span class="fa-fw play-button"></span></button>\n\n    <div class="audio-player__bar audio-player__bar--progress js-audio--seek">\n      <div class="audio-player__bar-current"></div>\n    </div>\n\n    <div class="audio-player__timestamps">\n      <div class="audio-player__timestamp audio-player__timestamp--current"></div>\n      <div class="audio-player__timestamp-separator">/</div>\n      <div class="audio-player__timestamp audio-player__timestamp--total"></div>\n    </div>\n  ', e
            };

            class l {
                constructor(t) {
                    Object.defineProperty(this, "userPreferences", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "audio", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Audio
                    }), Object.defineProperty(this, "currentSlider", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "durationFormatted", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "'0:00'"
                    }), Object.defineProperty(this, "hasWorkingVolumeControl", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !0
                    }), Object.defineProperty(this, "hideMainPlayerTimeout", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: -1
                    }), Object.defineProperty(this, "mainPlayer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "observer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "pagePlayer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "playerNext", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "playerPrev", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "settingNavigation", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "state", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "paused"
                    }), Object.defineProperty(this, "timeFormat", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "minute_minimal"
                    }), Object.defineProperty(this, "url", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "checkVolumeSettings", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = this.audio.volume, t = .1 === e ? .2 : .1;
                            this.audio.volume = t, setTimeout(() => {
                                this.hasWorkingVolumeControl = this.audio.volume === t, this.audio.volume = e, this.syncVolumeDisplay()
                            }, 0)
                        }
                    }), Object.defineProperty(this, "ensurePagePlayerIsAttached", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null == this.pagePlayer || document.body.contains(this.pagePlayer) || (this.pagePlayer = void 0)
                        }
                    }), Object.defineProperty(this, "load", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.dataset.audioUrl;
                            if (null == t) throw new Error("Player is missing url");
                            this.audio.paused || this.stop(), this.setTime(0), this.pagePlayer = e, this.url = t, this.audio.setAttribute("src", t), this.audio.currentTime = 0, this.setState("loading");
                            const s = this.audio.play();
                            null == s || s.catch(e => {
                                if (l.ignoredErrors.includes(e.name)) return console.debug("playback failed:", e.name), void this.stop();
                                throw e
                            }), this.setNavigation()
                        }
                    }), Object.defineProperty(this, "nav", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.currentTarget;
                            t instanceof HTMLElement && ("prev" === t.dataset.audioNav && null != this.playerPrev ? this.load(this.playerPrev) : "next" === t.dataset.audioNav && null != this.playerNext && this.load(this.playerNext))
                        }
                    }), Object.defineProperty(this, "observePage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.ensurePagePlayerIsAttached();
                            const t = [], s = [], r = null == this.pagePlayer;
                            e.forEach(e => {
                                e.addedNodes.forEach(e => {
                                    if (e instanceof HTMLElement && (e instanceof HTMLAudioElement ? t.push(e) : t.push(...e.querySelectorAll("audio")), r)) if (e.classList.contains("js-audio--player")) s.push(e); else for (const t of [...e.querySelectorAll(".js-audio--player")]) t instanceof HTMLElement && s.push(t)
                                })
                            }), s.push(...this.replaceAudioElems(t)), this.reattachPagePlayer(s)
                        }
                    }), Object.defineProperty(this, "onClickPlay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            e.preventDefault();
                            const t = this.findPlayer(e.currentTarget);
                            if (null == t) throw new Error("couldn't find pagePlayer of the play button");
                            t === this.pagePlayer ? this.togglePlay() : this.load(t)
                        }
                    }), Object.defineProperty(this, "onDocumentReady", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (null == this.mainPlayer) {
                                const e = document.querySelector(".js-audio--main");
                                if (null == e) return void console.debug("page is missing main player placeholder");
                                this.mainPlayer = a(), e.replaceWith(this.mainPlayer), Object(r.g)(() => this.audio.muted = this.userPreferences.get("audio_muted")), Object(r.g)(() => this.audio.volume = this.userPreferences.get("audio_volume")), this.checkVolumeSettings(), this.syncState()
                            }
                            null == this.observer && (this.observer = new MutationObserver(this.observePage), this.observer.observe(document, {
                                childList: !0,
                                subtree: !0
                            })), this.replaceAudioElems(), this.reattachPagePlayer()
                        }
                    }), Object.defineProperty(this, "onEnded", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.stop(), null != this.playerNext && this.userPreferences.get("audio_autoplay") && this.load(this.playerNext)
                        }
                    }), Object.defineProperty(this, "onPause", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setState("paused")
                        }
                    }), Object.defineProperty(this, "onPlaying", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.setTimeFormat(), this.durationFormatted = Object(i.a)(this.audio.duration, this.timeFormat), this.setState("playing")
                        }
                    }), Object.defineProperty(this, "onSeekEnd", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.currentSlider = void 0;
                            const t = 1 === e.getPercentage() ? this.audio.duration - .01 : this.audio.duration * e.getPercentage();
                            this.setTime(t)
                        }
                    }), Object.defineProperty(this, "onSeekStart", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.currentTarget;
                            t instanceof HTMLElement && (null != this.pagePlayer && this.pagePlayer.contains(t) || null != this.mainPlayer && this.mainPlayer.contains(t)) && Number.isFinite(this.audio.duration) && 0 !== this.audio.duration && (this.currentSlider = n.a.start({
                                bar: t,
                                endCallback: this.onSeekEnd,
                                initialEvent: e
                            }))
                        }
                    }), Object.defineProperty(this, "onTimeupdate", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.audio.paused && this.syncProgress()
                        }
                    }), Object.defineProperty(this, "onVolumeChangeEnd", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.currentSlider = void 0, this.userPreferences.set("audio_volume", this.audio.volume)
                        }
                    }), Object.defineProperty(this, "onVolumeChangeMove", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.audio.volume = e.getPercentage()
                        }
                    }), Object.defineProperty(this, "onVolumeChangeStart", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const t = e.currentTarget;
                            t instanceof HTMLElement && (this.currentSlider = n.a.start({
                                bar: t,
                                endCallback: this.onVolumeChangeEnd,
                                initialEvent: e,
                                moveCallback: this.onVolumeChangeMove
                            }))
                        }
                    }), Object.defineProperty(this, "reattachPagePlayer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (this.ensurePagePlayerIsAttached(), null != this.url && null == this.pagePlayer) {
                                null == e && (e = [...document.querySelectorAll(".js-audio--player")]);
                                for (const t of e) if (t instanceof HTMLElement && t.dataset.audioUrl === this.url) {
                                    this.pagePlayer = t, this.syncState();
                                    break
                                }
                            }
                            this.setNavigation()
                        }
                    }), Object.defineProperty(this, "replaceAudioElem", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            var t, s;
                            const r = null !== (t = osu.presence(e.src)) && void 0 !== t ? t : osu.presence(null === (s = e.querySelector("source")) || void 0 === s ? void 0 : s.src);
                            if (null == r) throw new Error("audio element is missing src");
                            const n = o();
                            return n.dataset.audioUrl = r, n.dataset.audioState = "paused", e.replaceWith(n), n
                        }
                    }), Object.defineProperty(this, "replaceAudioElems", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => (null == e && (e = [...document.querySelectorAll("audio")]), e.map(this.replaceAudioElem))
                    }), Object.defineProperty(this, "setNavigation", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.settingNavigation || (this.settingNavigation = !0, window.setTimeout(() => {
                                var e;
                                this.playerNext = void 0, this.playerPrev = void 0;
                                const t = null === (e = this.pagePlayer) || void 0 === e ? void 0 : e.closest(".js-audio--group");
                                if (t instanceof HTMLElement) {
                                    const e = t.querySelectorAll(".js-audio--player");
                                    for (let t = 0; t < e.length; t++) if (e[t] === this.pagePlayer) {
                                        if (t > 0) {
                                            const s = e[t - 1];
                                            s instanceof HTMLElement && (this.playerPrev = s)
                                        }
                                        const s = e[t + 1];
                                        s instanceof HTMLElement && (this.playerNext = s);
                                        break
                                    }
                                }
                                null != this.mainPlayer && (this.mainPlayer.dataset.audioHasPrev = null == this.playerPrev ? "0" : "1", this.mainPlayer.dataset.audioHasNext = null == this.playerNext ? "0" : "1"), this.settingNavigation = !1
                            }))
                        }
                    }), Object.defineProperty(this, "setState", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.state = e, this.syncState(), window.clearTimeout(this.hideMainPlayerTimeout), "playing" === this.state || "loading" === this.state ? null != this.mainPlayer && (this.mainPlayer.dataset.audioVisible = "1") : this.hideMainPlayerTimeout = window.setTimeout(() => {
                                null != this.mainPlayer && (this.mainPlayer.dataset.audioVisible = "0")
                            }, 4e3)
                        }
                    }), Object.defineProperty(this, "setTime", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            this.audio.currentTime = e, this.syncProgress()
                        }
                    }), Object.defineProperty(this, "setTimeFormat", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.audio.duration < 600 ? this.timeFormat = "minute_minimal" : this.audio.duration < 3600 ? this.timeFormat = "minute" : this.audio.duration < 36e3 ? this.timeFormat = "hour_minimal" : this.timeFormat = "hour"
                        }
                    }), Object.defineProperty(this, "stop", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var e;
                            this.audio.pause(), null === (e = this.currentSlider) || void 0 === e || e.end(), this.audio.currentTime = 0, this.onPause()
                        }
                    }), Object.defineProperty(this, "syncProgress", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (this.audio.duration > 0) {
                                const e = this.audio.currentTime / this.audio.duration, t = e >= .5 ? "1" : "0",
                                    s = e.toString(), r = Object(i.a)(this.audio.currentTime, this.timeFormat);
                                this.updatePlayers(e => {
                                    e.style.setProperty("--current-time", r), e.style.setProperty("--progress", s), e.dataset.audioOver50 = t
                                })
                            }
                            this.audio.paused || requestAnimationFrame(this.syncProgress)
                        }
                    }), Object.defineProperty(this, "syncState", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.updatePlayers(e => {
                                e.dataset.audioAutoplay = this.userPreferences.get("audio_autoplay") ? "1" : "0", e.dataset.audioHasDuration = Number.isFinite(this.audio.duration) ? "1" : "0", e.dataset.audioState = this.state, e.dataset.audioTimeFormat = this.timeFormat, e.style.setProperty("--duration", this.durationFormatted)
                            }), this.syncProgress(), this.syncVolumeDisplay()
                        }
                    }), Object.defineProperty(this, "syncVolumeDisplay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null != this.mainPlayer && (this.mainPlayer.dataset.audioVolumeBarVisible = this.hasWorkingVolumeControl ? "1" : "0", this.mainPlayer.dataset.audioVolume = this.volumeIcon(), this.mainPlayer.style.setProperty("--volume", this.audio.volume.toString()))
                        }
                    }), Object.defineProperty(this, "toggleAutoplay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.userPreferences.set("audio_autoplay", !this.userPreferences.get("audio_autoplay")), this.syncState()
                        }
                    }), Object.defineProperty(this, "toggleMute", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.userPreferences.set("audio_muted", !this.userPreferences.get("audio_muted"))
                        }
                    }), Object.defineProperty(this, "togglePlay", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            null != this.url && (this.audio.paused ? this.audio.play() : this.audio.pause())
                        }
                    }), Object.defineProperty(this, "updatePlayers", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            [this.mainPlayer, this.pagePlayer].forEach(t => {
                                null != t && e(t)
                            })
                        }
                    }), Object.defineProperty(this, "volumeIcon", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => this.audio.muted ? "muted" : 0 === this.audio.volume ? "silent" : this.audio.volume < .4 ? "quiet" : "normal"
                    }), this.audio.volume = 0, this.audio.addEventListener("pause", this.onPause), this.audio.addEventListener("playing", this.onPlaying), this.audio.addEventListener("ended", this.onEnded), this.audio.addEventListener("timeupdate", this.onTimeupdate), this.audio.addEventListener("volumechange", this.syncVolumeDisplay), e(document).on("click", ".js-audio--play", this.onClickPlay), e(document).on("click", ".js-audio--main-play", this.togglePlay), e(document).on(n.a.startEvents, ".js-audio--seek", this.onSeekStart), e(document).on(n.a.startEvents, ".js-audio--volume", this.onVolumeChangeStart), e(document).on("click", ".js-audio--toggle-mute", this.toggleMute), e(document).on("click", ".js-audio--toggle-autoplay", this.toggleAutoplay), e(document).on("click", ".js-audio--nav", this.nav), e(document).on("turbolinks:load", this.onDocumentReady)
                }

                findPlayer(e) {
                    var t, s;
                    const r = null !== (s = null === (t = this.mainPlayer) || void 0 === t ? void 0 : t.contains(e)) && void 0 !== s && s ? this.pagePlayer : e.closest(".js-audio--player");
                    if (r instanceof HTMLElement) return r
                }
            }

            Object.defineProperty(l, "ignoredErrors", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: ["AbortError", "NotAllowedError", "NotSupportedError"]
            })
        }).call(this, s("5wds"))
    }, nN9y: function (e, t, s) {
        "use strict";
        var r = s("WLnA");
        const n = ["chat.channel.join", "chat.channel.part"];
        var i = s("tz7b"), a = s("205K"), o = s("rqcF"), l = s("2hxc");

        function c(e) {
            if ("chat.message.new" === e.event) return new l.a(e.data);
            if (function (e) {
                return null != e.event && n.includes(e.event)
            }(e)) switch (e.event) {
                case"chat.channel.join":
                    return new a.a(e.data);
                case"chat.channel.part":
                    return new o.a(e.data.channel_id)
            }
        }

        let u = class {
            handleDispatchAction(e) {
                if (!(e instanceof i.a)) return;
                const t = c(e.message);
                null != t && Object(r.a)(t)
            }
        };
        u = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], u);
        t.a = u
    }, nxXY: function (e, t, s) {
        "use strict";

        function r(e) {
            return `${e.objectType}-${e.objectId}-${e.category}`
        }

        function n(e) {
            return null != e.objectId && null != e.category ? null != e.id ? "notification" : "stack" : "type"
        }

        function i(e) {
            return {category: e.category, id: e.id, objectId: e.object_id, objectType: e.object_type}
        }

        function a(e) {
            return {category: e.category, id: e.id, object_id: e.objectId, object_type: e.objectType}
        }

        function o(e) {
            return `${e.objectType}-${e.objectId}-${e.category}-${e.id}`
        }

        s.d(t, "c", (function () {
            return r
        })), s.d(t, "b", (function () {
            return n
        })), s.d(t, "a", (function () {
            return i
        })), s.d(t, "d", (function () {
            return a
        })), s.d(t, "e", (function () {
            return o
        }))
    }, o70V: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return n
            }));
            var r = s("5b8Q");

            class n {
                constructor() {
                    Object.defineProperty(this, "handleCancel", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t => {
                            var s;
                            t.preventDefault(), e.publish("forum-post-input:clear", [t.target]);
                            const n = e(t.target).parents(".js-forum-post-edit--container");
                            n.html(null !== (s = n.attr("data-original-post")) && void 0 !== s ? s : "").attr("data-original-post", null), Object(r.a)()
                        }
                    }), Object.defineProperty(this, "handleEditSaved", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            const s = e.target;
                            setTimeout(() => {
                                this.saved(s, t)
                            })
                        }
                    }), Object.defineProperty(this, "handleEditStart", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (e, t) => {
                            const s = e.target;
                            setTimeout(() => {
                                this.start(s, t)
                            })
                        }
                    }), Object.defineProperty(this, "saved", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            if (!(t instanceof HTMLElement)) throw new Error("target must be instance of HTMLElement");
                            e(t).parents(".js-forum-post").replaceWith(s), Object(r.a)()
                        }
                    }), Object.defineProperty(this, "start", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: (t, s) => {
                            if (!(t instanceof HTMLElement)) throw new Error("target must be instance of HTMLElement");
                            const n = e(t).parents(".js-forum-post-edit--container");
                            n.attr("data-original-post", n.html()).html(s).find("[name=body]").focus(), e.publish("forum-post-input:restore", [n[0]]), Object(r.a)()
                        }
                    }), e(document).on("ajax:success", ".js-edit-post-start", this.handleEditStart).on("click", ".js-edit-post-cancel", this.handleCancel).on("ajax:success", ".js-forum-post-edit", this.handleEditSaved)
                }
            }
        }).call(this, s("5wds"))
    }, oQBk: function (e, t, s) {
        "use strict";
        var r = s("/G9H");
        const n = Object(r.createContext)({isFriendsPage: !1});
        t.a = n
    }, oTtm: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("/G9H"), n = s("yun6"), i = s("tX/w");

        function a(e) {
            return r.createElement("div", {
                className: Object(i.a)("difficulty-badge", {"expert-plus": e.rating >= 6.5}),
                style: {"--bg": Object(n.d)(e.rating)}
            }, r.createElement("span", {className: "difficulty-badge__icon"}, r.createElement("span", {className: "fas fa-star"})), osu.formatNumber(e.rating, 2))
        }
    }, oWqU: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            }));
            var r = s("0h6b"), n = s("f4vq"), i = s("/G9H"), a = s("tX/w"), o = s("cX0L"), l = s("/HbY");

            class c extends i.PureComponent {
                constructor(t) {
                    super(t), Object.defineProperty(this, "state", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "eventId", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: `follow-toggle-${Object(o.a)()}`
                    }), Object.defineProperty(this, "toggleXhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: null
                    }), Object.defineProperty(this, "onClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var t;
                            const s = {
                                follow: {
                                    notifiable_id: this.props.follow.notifiable_id,
                                    notifiable_type: this.props.follow.notifiable_type,
                                    subtype: this.props.follow.subtype
                                }
                            }, n = this.state.following ? "DELETE" : "POST";
                            null === (t = this.toggleXhr) || void 0 === t || t.abort(), this.setState({toggling: !0}, () => {
                                this.toggleXhr = e.ajax(Object(r.a)("follows.store"), {data: s, method: n}).done(() => {
                                    "mapping" === this.props.follow.subtype ? e.publish("user:followUserMapping:update", {
                                        following: !this.state.following,
                                        userId: this.props.follow.notifiable_id
                                    }) : this.setState({following: !this.state.following})
                                }).always(() => {
                                    this.setState({toggling: !1})
                                })
                            })
                        }
                    }), Object.defineProperty(this, "refresh", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            "mapping" === this.props.follow.subtype && this.setState({following: null != n.a.currentUser && n.a.currentUser.follow_user_mapping.includes(this.props.follow.notifiable_id)})
                        }
                    }), this.state = {following: this.props.following, toggling: !1}
                }

                componentDidMount() {
                    "mapping" === this.props.follow.subtype && e.subscribe(`user:followUserMapping:refresh.${this.eventId}`, this.refresh)
                }

                componentWillUnmount() {
                    e.unsubscribe(`.${this.eventId}`)
                }

                render() {
                    return i.createElement("button", {
                        className: Object(a.a)("btn-circle", this.props.modifiers),
                        disabled: this.state.toggling,
                        onClick: this.onClick,
                        type: "button"
                    }, i.createElement("span", {className: "btn-circle__content"}, this.renderToggleIcon()))
                }

                renderToggleIcon() {
                    if (this.state.toggling) return i.createElement("span", {className: "btn-circle__icon"}, i.createElement(l.a, null));
                    let e, t;
                    return this.state.following ? (t = "fas fa-bell", e = "fas fa-bell-slash") : (t = "far fa-bell", e = "fas fa-bell"), i.createElement(i.Fragment, null, i.createElement("span", {className: "btn-circle__icon btn-circle__icon--hover-show"}, i.createElement("span", {className: e})), i.createElement("span", {className: "btn-circle__icon btn-circle__icon--hover-hide"}, i.createElement("span", {className: t})))
                }
            }

            Object.defineProperty(c, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {following: !0}
            })
        }).call(this, s("5wds"))
    }, pKj0: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        s("/G9H");
        var r, n = s("I8Ok");
        r = "beatmap-discussion-message-length-counter";
        var i = function (e) {
            var t, s, i;
            return i = e.message, e.isTimeline ? (s = BeatmapDiscussionHelper.MAX_LENGTH_TIMELINE, t = r, i.length > s ? t += " " + r + "--over" : i.length > .95 * s && (t += " " + r + "--almost-over"), Object(n.div)({className: t}, i.length + " / " + s)) : null
        }
    }, pTWL: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return o
            }));
            var n = s("lv9K"), i = s("WKXC"), a = s("is6n");

            class o {
                constructor(t) {
                    Object.defineProperty(this, "turbolinksReload", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: t
                    }), Object.defineProperty(this, "components", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "newVisit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !0
                    }), Object.defineProperty(this, "pageReady", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "renderedContainers", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Set
                    }), Object.defineProperty(this, "scrolled", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "timeoutScroll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "boot", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (this.pageReady && null != window.newBody) for (const [e, t] of this.components.entries()) {
                                const s = window.newBody.querySelectorAll(`.js-react--${e}`);
                                for (const e of s) e instanceof HTMLElement && !this.renderedContainers.has(e) && (this.renderedContainers.add(e), Object(n.u)(() => {
                                    i.render(t(e), e)
                                }))
                            }
                        }
                    }), Object.defineProperty(this, "destroy", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            for (const e of this.renderedContainers.values()) document.body.contains(e) || (i.unmountComponentAtNode(e), this.renderedContainers.delete(e))
                        }
                    }), Object.defineProperty(this, "handleBeforeCache", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.pageReady = !1, window.clearTimeout(this.timeoutScroll)
                        }
                    }), Object.defineProperty(this, "handleBeforeRender", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            window.newBody = e.originalEvent.data.newBody, this.setNewUrl(), this.pageReady = !0, this.loadScripts(!1), this.boot()
                        }
                    }), Object.defineProperty(this, "handleBeforeVisit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.newVisit = !0
                        }
                    }), Object.defineProperty(this, "handleLoad", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var t;
                            null !== (t = window.newBody) && void 0 !== t || (window.newBody = document.body), window.newUrl = null, this.pageReady = !0, this.scrolled = !1, e(window).off("scroll", this.handleWindowScroll), e(window).on("scroll", this.handleWindowScroll), window.setTimeout(() => {
                                this.destroy(), this.loadScripts(), this.boot(), this.timeoutScroll = window.setTimeout(this.scrollOnNewVisit, 100)
                            }, 1)
                        }
                    }), Object.defineProperty(this, "handleWindowScroll", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.scrolled = this.scrolled || 0 !== window.scrollX || 0 !== window.scrollY
                        }
                    }), Object.defineProperty(this, "scrollOnNewVisit", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            var t;
                            e(window).off("scroll", this.handleWindowScroll);
                            const s = this.newVisit;
                            if (this.newVisit = !1, !s || this.scrolled) return;
                            const r = decodeURIComponent(Object(a.a)().hash.substr(1));
                            "" !== r && (null === (t = document.getElementById(r)) || void 0 === t || t.scrollIntoView())
                        }
                    }), e(document).on("turbolinks:before-cache", this.handleBeforeCache), e(document).on("turbolinks:before-visit", this.handleBeforeVisit), e(document).on("turbolinks:load", this.handleLoad), e(document).on("turbolinks:before-render", this.handleBeforeRender)
                }

                register(e, t) {
                    this.components.has(e) || (this.components.set(e, t), this.boot())
                }

                runAfterPageLoad(t) {
                    if (document.body !== window.newBody) return e(document).one("turbolinks:load", t), () => {
                        e(document).off("turbolinks:load", t)
                    };
                    t()
                }

                loadScripts(e = !0) {
                    if (null == window.newBody) return;
                    const t = e ? "load" : "loadSync";
                    window.newBody.querySelectorAll(".js-react-turbolinks--script").forEach(e => {
                        if (e instanceof HTMLDivElement) {
                            const s = e.dataset.src;
                            null != s && this.turbolinksReload[t](s)
                        }
                    })
                }

                setNewUrl() {
                    var e, t, s, n;
                    const i = null !== (s = null === (t = null === (e = r.controller.currentVisit) || void 0 === e ? void 0 : e.redirectedToLocation) || void 0 === t ? void 0 : t.absoluteURL) && void 0 !== s ? s : null === (n = r.controller.currentVisit) || void 0 === n ? void 0 : n.location.absoluteURL;
                    window.newUrl = null == i ? document.location : new URL(i)
                }
            }
        }).call(this, s("5wds"), s("dMdw"))
    }, pdUJ: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("/G9H"), n = s("phBA"), i = function (e, t) {
            var s = {};
            for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (s[r] = e[r]);
            if (null != e && "function" == typeof Object.getOwnPropertySymbols) {
                var n = 0;
                for (r = Object.getOwnPropertySymbols(e); n < r.length; n++) t.indexOf(r[n]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[n]) && (s[r[n]] = e[r[n]])
            }
            return s
        };

        function a(e) {
            e.currentTarget.style.display = "none"
        }

        function o(e) {
            var t;
            const {hideOnError: s = !1} = e, o = i(e, ["hideOnError"]);
            return s && (o.onError = a), null == o.src ? (null !== (t = o.className) && void 0 !== t || (o.className = ""), o.className += " u-hidden", r.createElement("img", Object.assign({}, o))) : r.createElement("img", Object.assign({srcSet: `${o.src} 1x, ${Object(n.f)(o.src)} 2x`}, o))
        }
    }, phBA: function (e, t, s) {
        "use strict";
        (function (e) {
            function r() {
                return 0 === n()
            }

            function n() {
                const e = document.documentElement;
                return e.scrollHeight - e.scrollTop - e.clientHeight
            }

            function i(t) {
                if (t instanceof HTMLElement) return t instanceof HTMLFormElement ? () => e(t).submit() : () => t.click()
            }

            function a(e, t, s) {
                if (null == e) return;
                const r = ["", "k", "m", "b", "t"],
                    n = e => (null != s || (s = {}), null != t && (s.minimumFractionDigits = t, s.maximumFractionDigits = t), e.toLocaleString("en", s));
                if (e < 1e3) return n(e);
                const i = Math.min(r.length - 1, Math.floor(Math.log(e) / Math.log(1e3)));
                return `${n(e / Math.pow(1e3, i))}${r[i]}`
            }

            function o(e) {
                return e instanceof HTMLElement ? e : null
            }

            s.d(t, "a", (function () {
                return r
            })), s.d(t, "b", (function () {
                return n
            })), s.d(t, "c", (function () {
                return i
            })), s.d(t, "d", (function () {
                return a
            })), s.d(t, "e", (function () {
                return o
            })), s.d(t, "h", (function () {
                return l
            })), s.d(t, "f", (function () {
                return c
            })), s.d(t, "g", (function () {
                return u
            }));
            const l = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

            function c(e) {
                if (null != e) return e.replace(/(\.[^.]+)$/, "@2x$1")
            }

            function u(e) {
                return e.replace(/<[^>]*>/g, "")
            }
        }).call(this, s("5wds"))
    }, rLgU: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        })), s.d(t, "g", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        })), s.d(t, "c", (function () {
            return o
        })), s.d(t, "d", (function () {
            return l
        })), s.d(t, "h", (function () {
            return u
        })), s.d(t, "f", (function () {
            return p
        })), s.d(t, "e", (function () {
            return m
        }));
        var r = s("tSlR");
        const n = e => e.length,
            i = e => 0 === e.length || 1 === e.length && "paragraph" === e[0].type && 1 === e[0].children.length && "" === e[0].children[0].text,
            a = e => {
                var t;
                return "embed" === (null === (t = c(e)) || void 0 === t ? void 0 : t.type)
            }, o = e => {
                const t = c(e);
                return !!t && r.a.isEmpty(e, t)
            }, l = (e, t) => {
                const [s] = r.a.nodes(e, {match: e => !0 === e[t], mode: "all"});
                return !!s
            }, c = e => {
                if (e.selection) return r.c.parent(e, r.f.start(e.selection).path)
            }, u = (e, t) => {
                r.h.setNodes(e, {[t]: !l(e, t) || null}, {match: e => r.g.isText(e), split: !0})
            };

        function d(e) {
            const t = [], s = {bold: !1, italic: !1};
            return e.children.forEach(e => {
                var r, n, i, a;
                "" !== e.text && (s.bold !== (null !== (r = e.bold) && void 0 !== r && r) && (s.bold = null !== (n = e.bold) && void 0 !== n && n, t.push("**")), s.italic !== (null !== (i = e.italic) && void 0 !== i && i) && (s.italic = null !== (a = e.italic) && void 0 !== a && a, t.push("*"))), t.push(e.text.replace("*", "\\*"))
            }), s.bold && t.push("**"), s.italic && t.push("*"), t.join("")
        }

        const p = e => e.some(e => "embed" === e.type && "problem" === e.discussionType && !e.discussionId), m = e => {
            const t = [];
            e.forEach(e => {
                switch (e.type) {
                    case"paragraph":
                        t.push({text: d(e), type: "paragraph"});
                        break;
                    case"embed":
                        t.push(function (e) {
                            return e.discussionId ? {
                                discussion_id: e.discussionId,
                                type: "embed"
                            } : {
                                beatmap_id: e.beatmapId,
                                discussion_type: e.discussionType,
                                text: e.children[0].text,
                                timestamp: e.timestamp ? BeatmapDiscussionHelper.parseTimestamp(e.timestamp) : null,
                                type: "embed"
                            }
                        }(e))
                }
            });
            const s = t[t.length - 1];
            return "paragraph" !== s.type || osu.present(s.text) || t.pop(), JSON.stringify(t)
        }
    }, rMK6: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("Hs9Z"), n = s("/G9H"), i = s("tX/w");
        const a = e => {
            const t = Object(i.a)("supporter-icon", e.modifiers);
            return n.createElement("span", {
                className: t,
                title: osu.trans("users.show.is_supporter")
            }, Object(r.times)(e.level || 1, e => n.createElement("span", {key: e, className: "fas fa-heart"})))
        }
    }, rqcF: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("mkgZ");

        class n extends r.a {
            constructor(e) {
                super(), Object.defineProperty(this, "channelId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                })
            }
        }
    }, sBUI: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return w
            }));
            var n, i, a, o = s("4QOX"), l = s("f4vq"), c = s("/G9H"), u = s("I8Ok"), d = s("tX/w"), p = s("HBBY"),
                m = s("vMSe"), h = s("i41Q"), b = s("BzZm"), f = s("5eFc"), v = s("/HbY"), g = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, y = {}.hasOwnProperty;
            n = c.createElement, i = l.a.dataStore.commentStore, a = l.a.dataStore.uiState;
            var w = function (t) {
                function s() {
                    return this.renderFollowToggle = g(this.renderFollowToggle, this), this.renderShowDeletedToggle = g(this.renderShowDeletedToggle, this), this.renderComments = g(this.renderComments, this), this.renderComment = g(this.renderComment, this), this.render = g(this.render, this), s.__super__.constructor.apply(this, arguments)
                }

                return function (e, t) {
                    for (var s in t) y.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.render = function () {
                    return n(o.a, null, (t = this, function () {
                        var s, r;
                        return s = a.comments.topLevelCommentIds.map((function (e) {
                            return i.comments.get(e)
                        })), r = a.comments.pinnedCommentIds.map((function (e) {
                            return i.comments.get(e)
                        })), Object(u.div)({
                            className: Object(d.a)("comments", t.props.modifiers),
                            id: "comments"
                        }, Object(u.h2)({className: "comments__title"}, osu.trans("comments.title"), Object(u.span)({className: "comments__count"}, osu.formatNumber(a.comments.total))), r.length > 0 ? Object(u.div)({className: "comments__items comments__items--pinned"}, t.renderComments(r, !0)) : void 0, Object(u.div)({className: "comments__new"}, n(m.a, {
                            commentableType: t.props.commentableType,
                            commentableId: t.props.commentableId,
                            focus: !1,
                            modifiers: t.props.modifiers
                        })), Object(u.div)({className: "comments__items comments__items--toolbar"}, n(b.a, {modifiers: t.props.modifiers}), Object(u.div)({className: Object(d.a)("sort", t.props.modifiers)}, Object(u.div)({className: "sort__items"}, t.renderFollowToggle(), t.renderShowDeletedToggle()))), s.length > 0 ? Object(u.div)({className: "comments__items " + (null != a.comments.loadingSort ? "comments__items--loading" : "")}, t.renderComments(s, !1), n(f.a, {
                            comments: s,
                            modifiers: ["top"]
                        }), n(h.a, {
                            commentableType: t.props.commentableType,
                            commentableId: t.props.commentableId,
                            comments: s,
                            total: a.comments.topLevelCount,
                            sort: a.comments.currentSort,
                            modifiers: e.concat("top", t.props.modifiers)
                        })) : Object(u.div)({className: "comments__items comments__items--empty"}, osu.trans("comments.empty")))
                    }));
                    var t
                }, s.prototype.renderComment = function (e, t) {
                    return null == t && (t = !1), e.isDeleted && !l.a.userPreferences.get("comments_show_deleted") ? null : n(p.a, {
                        key: e.id,
                        comment: e,
                        depth: 0,
                        modifiers: this.props.modifiers,
                        expandReplies: !t && null
                    })
                }, s.prototype.renderComments = function (e, t) {
                    var s, r, n, i;
                    for (i = [], r = 0, n = e.length; r < n; r++) (s = e[r]).pinned === t && i.push(this.renderComment(s, t));
                    return i
                }, s.prototype.renderShowDeletedToggle = function () {
                    return Object(u.button)({
                        type: "button",
                        className: "sort__item sort__item--button",
                        onClick: this.toggleShowDeleted
                    }, Object(u.span)({className: "sort__item-icon"}, Object(u.span)({className: l.a.userPreferences.get("comments_show_deleted") ? "fas fa-check-square" : "far fa-square"})), osu.trans("common.buttons.show_deleted"))
                }, s.prototype.renderFollowToggle = function () {
                    var e, t, s;
                    return a.comments.userFollow ? (e = "fas fa-eye-slash", s = osu.trans("common.buttons.watch.to_0")) : (e = "fas fa-eye", s = osu.trans("common.buttons.watch.to_1")), t = this.props.loadingFollow ? n(v.a, {modifiers: ["center-inline"]}) : Object(u.span)({className: e}), Object(u.button)({
                        type: "button",
                        className: "sort__item sort__item--button",
                        onClick: this.toggleFollow,
                        disabled: this.props.loadingFollow
                    }, Object(u.span)({className: "sort__item-icon"}, t), s)
                }, s.prototype.toggleShowDeleted = function () {
                    return l.a.userPreferences.set("comments_show_deleted", !l.a.userPreferences.get("comments_show_deleted"))
                }, s.prototype.toggleFollow = function () {
                    return r.publish("comments:toggle-follow")
                }, s
            }(c.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, sHNI: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));
        const r = {
            hype: "fas fa-bullhorn",
            mapper_note: "far fa-sticky-note",
            praise: "fas fa-heart",
            problem: "fas fa-exclamation-circle",
            review: "fas fa-tasks",
            suggestion: "far fa-circle"
        }
    }, sTr9: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return d
        }));
        var r = s("bfq+"), n = s("Rfpg"), i = s("Hs9Z"), a = s("lv9K"), o = s("f4vq"), l = s("d+cC"),
            c = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            }, u = function (e, t, s, r) {
                return new (s || (s = Promise))((function (n, i) {
                    function a(e) {
                        try {
                            l(r.next(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function o(e) {
                        try {
                            l(r.throw(e))
                        } catch (t) {
                            i(t)
                        }
                    }

                    function l(e) {
                        var t;
                        e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                            e(t)
                        }))).then(a, o)
                    }

                    l((r = r.apply(e, t || [])).next())
                }))
            };

        class d {
            constructor(e) {
                Object.defineProperty(this, "canMessageError", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "channelId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "description", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "firstMessageId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: -1
                }), Object.defineProperty(this, "icon", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "inputText", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), Object.defineProperty(this, "lastReadId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "loadingEarlierMessages", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "loadingMessages", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "name", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), Object.defineProperty(this, "needsRefresh", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !0
                }), Object.defineProperty(this, "newPmChannel", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "type", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: "TEMPORARY"
                }), Object.defineProperty(this, "uiState", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: {autoScroll: !0, scrollY: 0}
                }), Object.defineProperty(this, "users", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: []
                }), Object.defineProperty(this, "messagesMap", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: new Map
                }), Object.defineProperty(this, "serverLastMessageId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), this.channelId = e, Object(a.p)(this)
            }

            get canMessage() {
                return null == this.canMessageError
            }

            get firstMessage() {
                return this.messages.length > 0 ? this.messages[0] : void 0
            }

            get hasEarlierMessages() {
                return this.firstMessageId !== this.minMessageId
            }

            get isDisplayable() {
                return this.name.length > 0 && null != this.icon
            }

            get isUnread() {
                return null != this.lastReadId ? this.lastMessageId > this.lastReadId : this.lastMessageId > -1
            }

            get lastMessage() {
                return this.messages[this.messages.length - 1]
            }

            get lastMessageId() {
                var e;
                for (let t = this.messages.length - 1; t >= 0; t--) if ("number" == typeof this.messages[t].messageId) return this.messages[t].messageId;
                return null !== (e = this.serverLastMessageId) && void 0 !== e ? e : -1
            }

            get messages() {
                return Object(i.sortBy)([...this.messagesMap.values()], ["timestamp", "channelId"])
            }

            get minMessageId() {
                const e = this.messages.length > 0 ? this.messages[0].messageId : void 0;
                return "number" == typeof e ? e : -1
            }

            get pmTarget() {
                if ("PM" === this.type) return this.users.find(e => e !== o.a.currentUserOrFail.id)
            }

            get supportedType() {
                return n.c.has(this.type) ? this.type : null
            }

            static newPM(e, t) {
                const s = new d(null != t ? t : -1);
                return s.newPmChannel = null == t, s.type = "PM", s.name = e.username, s.icon = e.avatarUrl, s.users = [o.a.currentUserOrFail.id, e.id], s
            }

            addMessage(e) {
                var t;
                if (null != e.uuid && e.sender_id === (null === (t = o.a.currentUser) || void 0 === t ? void 0 : t.id)) {
                    const t = this.messagesMap.get(e.uuid);
                    if (null != t) return this.persistMessage(t, e)
                }
                const s = l.a.fromJson(e);
                this.messagesMap.set(s.messageId, s)
            }

            addMessages(e) {
                e.forEach(e => this.messagesMap.set(e.messageId, e))
            }

            addSendingMessage(e) {
                this.messagesMap.set(e.messageId, e), this.markAsRead()
            }

            afterSendMesssage(e, t) {
                null != t ? (this.persistMessage(e, t), this.setLastReadId(t.message_id)) : e.errored = !0
            }

            load() {
                this.newPmChannel || this.refreshMessages()
            }

            loadEarlierMessages() {
                return u(this, void 0, void 0, (function* () {
                    if (!this.hasEarlierMessages || this.loadingEarlierMessages) return;
                    let e;
                    this.loadingEarlierMessages = !0, this.minMessageId > 0 && (e = this.minMessageId);
                    try {
                        const t = yield Object(r.c)(this.channelId, {until: e});
                        Object(a.u)(() => {
                            this.addMessages(t), 0 === t.length && (this.firstMessageId = this.minMessageId)
                        })
                    } finally {
                        Object(a.u)(() => {
                            this.loadingEarlierMessages = !1
                        })
                    }
                }))
            }

            markAsRead() {
                this.setLastReadId(this.lastMessageId)
            }

            refresh() {
                Object(r.b)(this.channelId).done(e => {
                    this.updateWithJson(e.channel)
                })
            }

            removeMessagesFromUserIds(e) {
                for (const [, t] of this.messagesMap) e.has(t.senderId) && this.messagesMap.delete(t.messageId)
            }

            setInputText(e) {
                this.inputText = e
            }

            updateWithJson(e) {
                var t, s;
                this.name = e.name, this.description = e.description, this.type = e.type, this.icon = null !== (t = e.icon) && void 0 !== t ? t : d.defaultIcon, this.users = null !== (s = e.users) && void 0 !== s ? s : this.users, this.serverLastMessageId = e.last_message_id, null != e.current_user_attributes && (this.canMessageError = e.current_user_attributes.can_message_error, this.setLastReadId(e.current_user_attributes.last_read_id))
            }

            persistMessage(e, t) {
                null != t.uuid && this.messagesMap.delete(t.uuid), e.persist(t), this.messagesMap.set(e.messageId, e)
            }

            refreshMessages() {
                return u(this, void 0, void 0, (function* () {
                    if (!this.needsRefresh || this.loadingMessages) return;
                    let e;
                    this.loadingMessages = !0, this.messages.length > 0 && this.lastMessageId > 0 && (e = this.lastMessageId);
                    try {
                        const t = yield Object(r.c)(this.channelId);
                        Object(a.u)(() => {
                            var s, r;
                            (null !== (r = null === (s = Object(i.minBy)(t, "messageId")) || void 0 === s ? void 0 : s.messageId) && void 0 !== r ? r : -1) > this.lastMessageId && this.messagesMap.clear(), this.addMessages(t), this.needsRefresh = !1, this.loadingMessages = !1, 0 !== t.length || null != e || (this.firstMessageId = this.minMessageId)
                        })
                    } catch (t) {
                        Object(a.u)(() => this.loadingMessages = !1)
                    }
                }))
            }

            setLastReadId(e) {
                var t;
                e > (null !== (t = this.lastReadId) && void 0 !== t ? t : 0) && (this.lastReadId = e)
            }
        }

        Object.defineProperty(d, "defaultIcon", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: "/images/layout/chat/channel-default.png"
        }), c([a.q], d.prototype, "canMessageError", void 0), c([a.q], d.prototype, "channelId", void 0), c([a.q], d.prototype, "description", void 0), c([a.q], d.prototype, "firstMessageId", void 0), c([a.q], d.prototype, "icon", void 0), c([a.q], d.prototype, "inputText", void 0), c([a.q], d.prototype, "lastReadId", void 0), c([a.q], d.prototype, "loadingEarlierMessages", void 0), c([a.q], d.prototype, "loadingMessages", void 0), c([a.q], d.prototype, "name", void 0), c([a.q], d.prototype, "newPmChannel", void 0), c([a.q], d.prototype, "type", void 0), c([a.q], d.prototype, "uiState", void 0), c([a.q], d.prototype, "users", void 0), c([a.q], d.prototype, "messagesMap", void 0), c([a.h], d.prototype, "canMessage", null), c([a.h], d.prototype, "firstMessage", null), c([a.h], d.prototype, "hasEarlierMessages", null), c([a.h], d.prototype, "isDisplayable", null), c([a.h], d.prototype, "isUnread", null), c([a.h], d.prototype, "lastMessage", null), c([a.h], d.prototype, "lastMessageId", null), c([a.h], d.prototype, "messages", null), c([a.h], d.prototype, "minMessageId", null), c([a.h], d.prototype, "pmTarget", null), c([a.h], d.prototype, "supportedType", null), c([a.f], d.prototype, "addMessage", null), c([a.f], d.prototype, "addMessages", null), c([a.f], d.prototype, "addSendingMessage", null), c([a.f], d.prototype, "afterSendMesssage", null), c([a.f], d.prototype, "load", null), c([a.f], d.prototype, "loadEarlierMessages", null), c([a.f], d.prototype, "markAsRead", null), c([a.f], d.prototype, "refresh", null), c([a.f], d.prototype, "removeMessagesFromUserIds", null), c([a.f], d.prototype, "setInputText", null), c([a.f], d.prototype, "updateWithJson", null), c([a.f], d.prototype, "persistMessage", null), c([a.f], d.prototype, "refreshMessages", null), c([a.f], d.prototype, "setLastReadId", null)
    }, srn7: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        }));
        var r = s("lv9K"), n = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class i {
            constructor(e) {
                Object.defineProperty(this, "avatarUrl", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: "/images/layout/avatar-guest.png"
                }), Object.defineProperty(this, "countryCode", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: "XX"
                }), Object.defineProperty(this, "defaultGroup", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), Object.defineProperty(this, "groups", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "id", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "isActive", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isBot", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isDeleted", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isOnline", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isSupporter", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "lastVisit", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: null
                }), Object.defineProperty(this, "loaded", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "pmFriendsOnly", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "profileColour", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), Object.defineProperty(this, "username", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: ""
                }), this.id = e, Object(r.p)(this)
            }

            static fromJson(e) {
                const t = new i(e.id);
                return Object.assign(t, {
                    avatarUrl: e.avatar_url,
                    countryCode: e.country_code,
                    defaultGroup: e.default_group,
                    groups: e.groups,
                    isActive: e.is_active,
                    isBot: e.is_bot,
                    isOnline: e.is_online,
                    isSupporter: e.is_supporter,
                    loaded: !0,
                    pmFriendsOnly: e.pm_friends_only,
                    profileColour: e.profile_colour,
                    username: e.username
                })
            }

            is(e) {
                return null != e && e.id === this.id
            }

            load() {
            }

            toJson() {
                return {
                    avatar_url: this.avatarUrl,
                    country_code: this.countryCode,
                    default_group: this.defaultGroup,
                    groups: this.groups,
                    id: this.id,
                    is_active: this.isActive,
                    is_bot: this.isBot,
                    is_deleted: this.isDeleted,
                    is_online: this.isOnline,
                    is_supporter: this.isSupporter,
                    last_visit: this.lastVisit,
                    pm_friends_only: this.pmFriendsOnly,
                    profile_colour: this.profileColour,
                    username: this.username
                }
            }

            updateFromJson(e) {
                var t;
                this.username = e.username, this.avatarUrl = e.avatar_url, this.profileColour = null !== (t = e.profile_colour) && void 0 !== t ? t : "", this.countryCode = e.country_code, this.isSupporter = e.is_supporter, this.isActive = e.is_active, this.isBot = e.is_bot, this.isOnline = e.is_online, this.pmFriendsOnly = e.pm_friends_only, this.loaded = !0
            }
        }

        n([r.q], i.prototype, "avatarUrl", void 0), n([r.q], i.prototype, "countryCode", void 0), n([r.q], i.prototype, "defaultGroup", void 0), n([r.q], i.prototype, "groups", void 0), n([r.q], i.prototype, "id", void 0), n([r.q], i.prototype, "isActive", void 0), n([r.q], i.prototype, "isBot", void 0), n([r.q], i.prototype, "isDeleted", void 0), n([r.q], i.prototype, "isOnline", void 0), n([r.q], i.prototype, "isSupporter", void 0), n([r.q], i.prototype, "lastVisit", void 0), n([r.q], i.prototype, "loaded", void 0), n([r.q], i.prototype, "pmFriendsOnly", void 0), n([r.q], i.prototype, "profileColour", void 0), n([r.q], i.prototype, "username", void 0), n([r.f], i.prototype, "updateFromJson", null);
        const a = new i(-1);
        a.isDeleted = !0, a.username = osu.trans("users.deleted")
    }, ss8h: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");
        const n = r.createContext(null)
    }, tGwB: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return a
            }));
            var r = s("iBbO"), n = s("/G9H");
            const i = "click-to-copy";

            class a extends n.Component {
                constructor() {
                    super(...arguments), Object.defineProperty(this, "linkRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: n.createRef()
                    }), Object.defineProperty(this, "timer", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "titles", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {
                            default: osu.trans("common.buttons.click_to_copy"),
                            onClick: osu.trans("common.buttons.click_to_copy_copied")
                        }
                    }), Object.defineProperty(this, "onClick", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            e.preventDefault(), r.writeText(this.props.value);
                            const t = this.api;
                            null != t && (t.set("content.text", this.titles.onClick), window.clearTimeout(this.timer), this.timer = window.setTimeout(this.resetTooltip, 1e3))
                        }
                    }), Object.defineProperty(this, "onMouseLeave", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            window.clearTimeout(this.timer), this.resetTooltip()
                        }
                    }), Object.defineProperty(this, "resetTooltip", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = this.api;
                            null != e && (e.hide(), this.timer = window.setTimeout(() => {
                                e.set("content.text", this.titles.default)
                            }, 100))
                        }
                    })
                }

                get api() {
                    if (null != this.linkRef.current) return e(this.linkRef.current).qtip("api")
                }

                componentWillUnmount() {
                    window.clearTimeout(this.timer)
                }

                render() {
                    return this.props.value ? n.createElement("a", {
                        ref: this.linkRef,
                        className: i,
                        "data-tooltip-hide-events": "mouseleave",
                        "data-tooltip-pin-position": !0,
                        "data-tooltip-position": "bottom center",
                        href: this.props.valueAsUrl ? this.props.value : "#",
                        onClick: this.onClick,
                        onMouseLeave: this.onMouseLeave,
                        title: this.titles.default
                    }, this.props.label || this.props.value, this.props.showIcon && n.createElement("i", {className: `fas fa-paste ${i}__icon`})) : n.createElement("span", null)
                }
            }

            Object.defineProperty(a, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {showIcon: !1, valueAsUrl: !1}
            })
        }).call(this, s("5wds"))
    }, "tX/w": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        })), s.d(t, "b", (function () {
            return a
        }));
        var r = s("Hs9Z");
        const n = (e, t) => {
            e.forEach(e => {
                Array.isArray(e) ? e.forEach(e => {
                    null != e && t(e)
                }) : "string" == typeof e ? t(e) : Object(r.forEach)(e, (e, s) => {
                    e && t(s)
                })
            })
        };

        function i(e, ...t) {
            let s = e;
            return n(t, t => s += ` ${e}--${t}`), s
        }

        function a(...e) {
            const t = [];
            return n(e, e => t.push(e)), t
        }
    }, tnRH: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");

        function n(e) {
            return r.createElement("div", {className: "u-relative"}, r.createElement("h2", {className: "title title--page-extra"}, osu.trans(`users.show.extra.${e.name}.title`)), e.withEdit && r.createElement("span", {className: "sortable-handle sortable-handle--profile-page-extra hidden-xs js-profile-page-extra--sortable-handle"}, r.createElement("i", {className: "fas fa-bars"})))
        }
    }, tz7b: function (e, t, s) {
        "use strict";
        s.d(t, "b", (function () {
            return n
        })), s.d(t, "a", (function () {
            return i
        }));
        var r = s("mkgZ");

        function n(e) {
            return "object" == typeof e && null != e && "event" in e
        }

        class i extends r.a {
            constructor(e) {
                super(), Object.defineProperty(this, "message", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                })
            }
        }
    }, "u/q5": function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        })), s.d(t, "b", (function () {
            return i
        })), s.d(t, "c", (function () {
            return a
        })), s.d(t, "d", (function () {
            return o
        })), s.d(t, "e", (function () {
            return c
        }));
        var r = s("f4vq");

        function n(e) {
            return null != e.best_id && !e.user.is_deleted && null != r.a.currentUser && e.user_id !== r.a.currentUser.id
        }

        function i(e) {
            return n(e) || a(e) || o(e) || r.a.scorePins.canBePinned(e)
        }

        function a(e) {
            return e.replay
        }

        function o(e) {
            return null != e.best_id
        }

        const l = osu.trans("beatmapsets.show.scoreboard.headers.miss"), c = {
            fruits: [{attribute: "count_300", label: "fruits"}, {
                attribute: "count_100",
                label: "ticks"
            }, {attribute: "count_katu", label: "drp miss"}, {attribute: "count_miss", label: l}],
            mania: [{attribute: "count_geki", label: "max"}, {
                attribute: "count_300",
                label: "300"
            }, {attribute: "count_katu", label: "200"}, {attribute: "count_100", label: "100"}, {
                attribute: "count_50",
                label: "50"
            }, {attribute: "count_miss", label: l}],
            osu: [{attribute: "count_300", label: "300"}, {
                attribute: "count_100",
                label: "100"
            }, {attribute: "count_50", label: "50"}, {attribute: "count_miss", label: l}],
            taiko: [{attribute: "count_300", label: "great"}, {
                attribute: "count_100",
                label: "good"
            }, {attribute: "count_miss", label: l}]
        }
    }, uP2m: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("f4vq"), n = s("/G9H"), i = s("cX0L");

        class a extends n.PureComponent {
            constructor() {
                super(...arguments), Object.defineProperty(this, "eventId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: `admin-menu-${Object(i.a)()}`
                })
            }

            render() {
                var e;
                if (!(null === (e = r.a.currentUser) || void 0 === e ? void 0 : e.is_admin)) return null;
                const t = this.props.items.map(e => n.createElement(e.component, Object.assign({
                    key: `${e.icon}-${e.text}`,
                    className: "admin-menu-item"
                }, e.props), n.createElement("span", {className: "admin-menu-item__content"}, n.createElement("span", {className: "admin-menu-item__label admin-menu-item__label--icon"}, n.createElement("span", {className: e.icon})), n.createElement("span", {className: "admin-menu-item__label admin-menu-item__label--text"}, e.text))));
                return n.createElement("div", {className: "admin-menu"}, n.createElement("button", {
                    className: "admin-menu__button js-menu",
                    "data-menu-target": `admin-menu-${this.eventId}`
                }, n.createElement("span", {className: "fas fa-angle-up"}), n.createElement("span", {className: "admin-menu__button-icon fas fa-tools"})), n.createElement("div", {
                    className: "admin-menu__menu js-menu",
                    "data-menu-id": `admin-menu-${this.eventId}`,
                    "data-visibility": "hidden"
                }, t))
            }
        }
    }, "uW+8": function (e, t, s) {
        "use strict";
        s.d(t, "b", (function () {
            return i
        })), s.d(t, "c", (function () {
            return a
        })), s.d(t, "a", (function () {
            return o
        })), s.d(t, "d", (function () {
            return l
        }));
        var r = s("mkgZ"), n = s("nxXY");

        class i extends r.a {
            constructor(e, t) {
                super(), Object.defineProperty(this, "data", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "context", {enumerable: !0, configurable: !0, writable: !0, value: t})
            }
        }

        class a extends r.a {
            constructor(e) {
                super(), Object.defineProperty(this, "data", {enumerable: !0, configurable: !0, writable: !0, value: e})
            }
        }

        class o extends r.a {
            constructor(e, t) {
                super(), Object.defineProperty(this, "data", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "readCount", {enumerable: !0, configurable: !0, writable: !0, value: t})
            }

            static fromJson(e) {
                const t = e.data.notifications.map(e => Object(n.a)(e));
                return new o(t, e.data.read_count)
            }
        }

        class l extends r.a {
            constructor(e, t) {
                super(), Object.defineProperty(this, "data", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "readCount", {enumerable: !0, configurable: !0, writable: !0, value: t})
            }

            static fromJson(e) {
                const t = e.data.notifications.map(e => Object(n.a)(e));
                return new l(t, e.data.read_count)
            }
        }
    }, ubBH: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            })), s.d(t, "b", (function () {
                return u
            })), s.d(t, "c", (function () {
                return d
            }));
            var r = s("LXy+"), n = s("0h6b"), i = s("Hs9Z"), a = s("lv9K"), o = s("f4vq"), l = s("/jJF");

            function c(e, t) {
                return Object(r.a)(e) ? e[t] : Object(i.sum)(Object.values(e[t]))
            }

            function u(e) {
                return !e.nsfw || o.a.userPreferences.get("beatmapset_show_nsfw")
            }

            const d = Object(a.f)(t => {
                const s = () => d(t);
                if (o.a.userLogin.showIfGuest(s)) return;
                const r = !t.has_favourited;
                t.has_favourited = r, t.favourite_count += r ? 1 : -1, e.ajax(Object(n.a)("beatmapsets.favourites.store", {beatmapset: t.id}), {
                    data: {action: r ? "favourite" : "unfavourite"},
                    method: "POST"
                }).fail(Object(a.f)((e, n) => {
                    t.has_favourited = !r, t.favourite_count += r ? -1 : 1, Object(l.a)(e, n, s)
                })).done(Object(a.f)(e => {
                    t.favourite_count = e.favourite_count
                }))
            })
        }).call(this, s("5wds"))
    }, ueqr: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return r
        }));
        const r = {
            audio_autoplay: !1,
            audio_muted: !1,
            audio_volume: .45,
            beatmapset_card_size: "normal",
            beatmapset_download: "all",
            beatmapset_show_nsfw: !1,
            beatmapset_title_show_original: !1,
            comments_show_deleted: !1,
            forum_posts_show_deleted: !0,
            profile_cover_expanded: !0,
            user_list_filter: "all",
            user_list_sort: "last_visit",
            user_list_view: "card"
        }
    }, ufn5: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return P
            }));
            var r = s("IwO5"), n = s("/HbY"), i = s("PQII"), a = s.n(i), o = s("0h6b"), l = s("Hs9Z"), c = s("f4vq"),
                u = s("/G9H"), d = s("tSlR"), p = s("21/t"), m = s("dTpI"), h = s("yun6"), b = s("ubBH"), f = s("tX/w"),
                v = s("5AHq"), g = s("BCxl"), y = s("rLgU"), w = s("/cS6"), _ = s("BTwX"), O = s("XMjx"), j = s("6b0J"),
                E = s("ss8h");

            class P extends u.Component {
                constructor(t) {
                    super(t), Object.defineProperty(this, "bn", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: "beatmap-discussion-editor"
                    }), Object.defineProperty(this, "cache", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {}
                    }), Object.defineProperty(this, "emptyDocTemplate", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: [{children: [{text: ""}], type: "paragraph"}]
                    }), Object.defineProperty(this, "insertMenuRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "localStorageKey", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "scrollContainerRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "slateEditor", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "toolbarRef", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "blockWrapper", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => u.createElement("div", {className: `${this.bn}__block`}, e)
                    }), Object.defineProperty(this, "decorateTimestamps", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const [t, s] = e, r = [];
                            if (!d.g.isText(t)) return r;
                            const n = RegExp(BeatmapDiscussionHelper.TIMESTAMP_REGEX, "g");
                            let i;
                            for (; null !== (i = n.exec(t.text));) r.push({
                                anchor: {offset: i.index, path: s},
                                focus: {offset: i.index + i[0].length, path: s},
                                timestamp: i[0]
                            });
                            return r
                        }
                    }), Object.defineProperty(this, "isCurrentBeatmap", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => null != e && e.beatmapset_id === this.props.beatmapset.id
                    }), Object.defineProperty(this, "onChange", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            if (0 === e.length && (e = this.emptyDocTemplate), !this.props.editMode) {
                                const t = JSON.stringify(e);
                                Object(y.g)(e) ? localStorage.removeItem(this.localStorageKey) : localStorage.setItem(this.localStorageKey, t)
                            }
                            this.setState({blockCount: Object(y.a)(e), value: e}, () => {
                                var e, t;
                                m.b.isFocused(this.slateEditor) && this.props.onFocus && this.props.onFocus(), null === (t = (e = this.props).onChange) || void 0 === t || t.call(e)
                            })
                        }
                    }), Object.defineProperty(this, "onKeyDown", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            a()("mod+b", e) ? (e.preventDefault(), Object(y.h)(this.slateEditor, "bold")) : a()("mod+i", e) ? (e.preventDefault(), Object(y.h)(this.slateEditor, "italic")) : a()("shift+enter", e) ? Object(y.b)(this.slateEditor) && (e.preventDefault(), this.slateEditor.insertText("\n")) : (a()("delete", e) || a()("backspace", e)) && Object(y.c)(this.slateEditor) && (e.preventDefault(), d.h.removeNodes(this.slateEditor))
                        }
                    }), Object.defineProperty(this, "post", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.showConfirmationIfRequired() && this.setState({posting: !0}, () => {
                                this.xhr = e.ajax(Object(o.a)("beatmapsets.discussion.review", {beatmapset: this.props.beatmapset.id}), {
                                    data: {document: this.serialize()},
                                    method: "POST"
                                }).done(t => {
                                    e.publish("beatmapsetDiscussions:update", {beatmapset: t}), this.resetInput()
                                }).fail(osu.ajaxError).always(() => this.setState({posting: !1}))
                            })
                        }
                    }), Object.defineProperty(this, "renderBlockCount", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => u.createElement(r.a, {
                            current: this.state.blockCount,
                            max: this.context.max_blocks,
                            onlyShowAsWarning: !0,
                            theme: e,
                            tooltip: osu.trans("beatmap_discussions.review.block_count", {
                                max: this.context.max_blocks,
                                used: this.state.blockCount
                            })
                        })
                    }), Object.defineProperty(this, "renderElement", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            let t;
                            switch (e.element.type) {
                                case"embed":
                                    t = u.createElement(g.a, Object.assign({
                                        beatmaps: this.sortedBeatmaps(),
                                        beatmapset: this.props.beatmapset,
                                        currentBeatmap: this.props.currentBeatmap,
                                        discussions: this.props.discussions,
                                        editMode: this.props.editMode,
                                        readOnly: this.state.posting
                                    }, e));
                                    break;
                                default:
                                    t = e.children
                            }
                            return this.blockWrapper(t)
                        }
                    }), Object.defineProperty(this, "renderLeaf", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            let t = e.children;
                            return e.leaf.bold && (t = u.createElement("strong", Object.assign({}, e.attributes), t)), e.leaf.italic && (t = u.createElement("em", Object.assign({}, e.attributes), t)), e.leaf.timestamp ? u.createElement("span", Object.assign({className: "beatmap-discussion-timestamp-decoration"}, e.attributes), t) : u.createElement("span", Object.assign({}, e.attributes), t)
                        }
                    }), Object.defineProperty(this, "resetInput", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            e && (e.preventDefault(), !confirm(osu.trans("common.confirmation"))) || (d.h.deselect(this.slateEditor), this.onChange(this.emptyDocTemplate))
                        }
                    }), Object.defineProperty(this, "serialize", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => Object(y.e)(this.state.value)
                    }), Object.defineProperty(this, "showConfirmationIfRequired", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            const e = Object(y.f)(this.state.value),
                                t = null != c.a.currentUser && (c.a.currentUser.is_admin || c.a.currentUser.is_moderator || c.a.currentUser.is_full_bn),
                                s = "qualified" === this.props.beatmapset.status && e,
                                r = null != c.a.currentUser && (c.a.currentUser.is_admin || c.a.currentUser.is_nat || c.a.currentUser.is_bng),
                                n = "pending" === this.props.beatmapset.status && this.props.beatmapset.nominations && Object(b.a)(this.props.beatmapset.nominations, "current") > 0 && e;
                            return t && s ? confirm(osu.trans("beatmaps.nominations.reset_confirm.disqualify")) : !r || !n || confirm(osu.trans("beatmaps.nominations.reset_confirm.nomination_reset"))
                        }
                    }), Object.defineProperty(this, "sortedBeatmaps", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            if (null == this.cache.sortedBeatmaps) {
                                const e = l.filter(this.props.beatmaps, this.isCurrentBeatmap);
                                this.cache.sortedBeatmaps = Object(h.h)(e)
                            }
                            return this.cache.sortedBeatmaps
                        }
                    }), Object.defineProperty(this, "updateDrafts", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.cache.draftEmbeds = this.state.value.filter(e => "embed" === e.type && !e.discussion_id)
                        }
                    }), Object.defineProperty(this, "withNormalization", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            const {insertData: t, normalizeNode: s} = e;
                            return e.insertData = s => {
                                Object(y.b)(this.slateEditor) ? e.insertText(s.getData("text/plain")) : t(s)
                            }, e.normalizeNode = t => {
                                const [r, n] = t;
                                if (d.b.isElement(r) && "embed" === r.type) for (const [s, i] of d.c.children(e, n)) {
                                    if (d.b.isElement(s) && !e.isInline(s)) return void d.h.unwrapNodes(e, {at: i});
                                    if (s.bold || s.italic) return void d.h.unsetNodes(e, ["bold", "italic"], {at: i});
                                    if (null != r.beatmapId) {
                                        const t = "number" == typeof r.beatmapId ? this.props.beatmaps[r.beatmapId] : void 0;
                                        null != t && null == t.deleted_at || d.h.setNodes(e, {beatmapId: void 0}, {at: n})
                                    }
                                }
                                s(t)
                            }, e
                        }
                    }), this.slateEditor = this.withNormalization(Object(p.a)(Object(m.d)(Object(d.i)()))), this.scrollContainerRef = u.createRef(), this.toolbarRef = u.createRef(), this.insertMenuRef = u.createRef(), this.localStorageKey = `newDiscussion-${this.props.beatmapset.id}`;
                    let s = this.emptyDocTemplate;
                    if (t.editMode) s = this.valueFromProps(); else {
                        const e = localStorage.getItem(this.localStorageKey);
                        if (null != e) try {
                            s = JSON.parse(e)
                        } catch (n) {
                            console.error("invalid json in localStorage, ignoring"), localStorage.removeItem(this.localStorageKey)
                        }
                    }
                    this.state = {blockCount: Object(y.a)(s), posting: !1, value: s}
                }

                get canSave() {
                    return !this.state.posting && this.state.blockCount <= this.context.max_blocks
                }

                componentDidMount() {
                    this.scrollContainerRef.current && (this.toolbarRef.current && this.toolbarRef.current.setScrollContainer(this.scrollContainerRef.current), this.insertMenuRef.current && this.insertMenuRef.current.setScrollContainer(this.scrollContainerRef.current))
                }

                componentDidUpdate(e) {
                    if (this.props.document !== e.document) {
                        const e = this.valueFromProps();
                        this.setState({blockCount: Object(y.a)(e), value: e})
                    }
                }

                componentWillUnmount() {
                    this.xhr && this.xhr.abort()
                }

                render() {
                    this.cache = {};
                    const e = "beatmap-discussion-editor", t = this.props.editMode ? ["edit-mode"] : [];
                    return this.state.posting && t.push("readonly"), this.updateDrafts(), u.createElement("div", {className: Object(f.a)(e, t)}, u.createElement("div", {className: `${e}__content`}, u.createElement(E.a.Provider, {value: this.slateEditor}, u.createElement(m.c, {
                        editor: this.slateEditor,
                        onChange: this.onChange,
                        value: this.state.value
                    }, u.createElement("div", {
                        ref: this.scrollContainerRef,
                        className: `${e}__input-area`
                    }, u.createElement(_.a, {ref: this.toolbarRef}), u.createElement(w.a, {
                        ref: this.insertMenuRef,
                        currentBeatmap: this.props.currentBeatmap
                    }), u.createElement(v.a.Provider, {value: this.cache.draftEmbeds || []}, u.createElement(m.a, {
                        decorate: this.decorateTimestamps,
                        onKeyDown: this.onKeyDown,
                        placeholder: osu.trans("beatmaps.discussions.message_placeholder.review"),
                        readOnly: this.state.posting,
                        renderElement: this.renderElement,
                        renderLeaf: this.renderLeaf
                    }))), this.props.editMode && u.createElement("div", {className: `${e}__inner-block-count`}, this.renderBlockCount("lighter")), !this.props.editMode && u.createElement("div", {className: `${e}__button-bar`}, u.createElement("button", {
                        className: "btn-osu-big btn-osu-big--forum-secondary",
                        disabled: this.state.posting,
                        onClick: this.resetInput,
                        type: "button"
                    }, osu.trans("common.buttons.clear")), u.createElement("div", null, u.createElement("span", {className: `${e}__block-count`}, this.renderBlockCount()), u.createElement("button", {
                        className: "btn-osu-big btn-osu-big--forum-primary",
                        disabled: !this.canSave,
                        onClick: this.post,
                        type: "submit"
                    }, this.state.posting ? u.createElement(n.a, null) : osu.trans("common.buttons.post"))))))))
                }

                valueFromProps() {
                    return this.props.editing && null != this.props.document && null != this.props.discussions ? Object(O.a)(this.props.document, this.props.discussions) : []
                }
            }

            Object.defineProperty(P, "contextType", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: j.a
            }), Object.defineProperty(P, "defaultProps", {
                enumerable: !0,
                configurable: !0,
                writable: !0,
                value: {editing: !1}
            })
        }).call(this, s("5wds"))
    }, uuPA: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return g
            }));
            var n, i, a = s("cQQh"), o = s("c1EF"), l = s("0h6b"), c = s("f4vq"), u = s("/G9H"), d = s("LQV2"),
                p = s.n(d), m = s("I8Ok"), h = (s("phBA"), s("Ma8u")), b = s("pKj0"), f = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, v = {}.hasOwnProperty;
            i = u.createElement, n = "beatmap-discussion-post";
            var g = function (t) {
                var s;

                function d(t) {
                    var s;
                    this.validPost = f(this.validPost, this), this.storedMessage = f(this.storedMessage, this), this.storeMessage = f(this.storeMessage, this), this.storageKey = f(this.storageKey, this), this.setMessage = f(this.setMessage, this), this.post = f(this.post, this), this.onCancelClick = f(this.onCancelClick, this), this.isTimeline = f(this.isTimeline, this), this.handleKeyDownCallback = f(this.handleKeyDownCallback, this), this.editStart = f(this.editStart, this), this.canResolve = f(this.canResolve, this), this.canReopen = f(this.canReopen, this), this.renderReplyButton = f(this.renderReplyButton, this), this.renderPlaceholder = f(this.renderPlaceholder, this), this.renderCancelButton = f(this.renderCancelButton, this), this.renderBox = f(this.renderBox, this), this.render = f(this.render, this), this.componentWillUnmount = f(this.componentWillUnmount, this), this.componentDidUpdate = f(this.componentDidUpdate, this), d.__super__.constructor.call(this, t), this.box = u.createRef(), this.throttledPost = e.throttle(this.post, 1e3), this.handleKeyDown = InputHandler.textarea(this.handleKeyDownCallback), s = this.storedMessage(), this.state = {
                        editing: "" !== s,
                        message: s,
                        posting: null
                    }
                }

                return function (e, t) {
                    for (var s in t) v.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(d, t), s = {
                    reply_resolve: "fas fa-check",
                    reply_reopen: "fas fa-exclamation-circle",
                    reply: "fas fa-reply"
                }, d.prototype.componentDidUpdate = function (e) {
                    if (e.discussion.id === this.props.discussion.id) return this.storeMessage();
                    this.setState({message: this.storedMessage()})
                }, d.prototype.componentWillUnmount = function () {
                    var e;
                    return this.throttledPost.cancel(), null != (e = this.postXhr) ? e.abort() : void 0
                }, d.prototype.render = function () {
                    return this.state.editing ? this.renderBox() : this.renderPlaceholder()
                }, d.prototype.renderBox = function () {
                    return Object(m.div)({className: n + " " + n + "--reply " + n + "--new-reply"}, this.renderCancelButton(), Object(m.div)({className: n + "__content"}, Object(m.div)({className: n + "__avatar"}, i(o.a, {
                        user: this.props.currentUser,
                        modifiers: ["full-rounded"]
                    })), Object(m.div)({className: n + "__message-container"}, i(p.a, {
                        disabled: null != this.state.posting,
                        className: n + "__message " + n + "__message--editor",
                        value: this.state.message,
                        onChange: this.setMessage,
                        onKeyDown: this.handleKeyDown,
                        placeholder: osu.trans("beatmaps.discussions.reply_placeholder"),
                        ref: this.box
                    }))), Object(m.div)({className: n + "__footer " + n + "__footer--notice"}, osu.trans("beatmaps.discussions.reply_notice"), i(b.a, {
                        message: this.state.message,
                        isTimeline: this.isTimeline()
                    })), Object(m.div)({className: n + "__footer"}, Object(m.div)({className: n + "__actions"}, Object(m.div)({className: n + "__actions-group"}, this.canResolve() && !this.props.discussion.resolved ? this.renderReplyButton("reply_resolve") : void 0, this.canReopen() && this.props.discussion.resolved ? this.renderReplyButton("reply_reopen") : void 0, this.renderReplyButton("reply")))))
                }, d.prototype.renderCancelButton = function () {
                    return Object(m.button)({
                        className: n + "__action " + n + "__action--cancel",
                        disabled: null != this.state.posting,
                        onClick: this.onCancelClick
                    }, Object(m.i)({className: "fas fa-times"}))
                }, d.prototype.renderPlaceholder = function () {
                    var e, t, s, r;
                    return r = (s = null != this.props.currentUser.id ? [osu.trans("beatmap_discussions.reply.open.user"), "fas fa-reply", this.props.currentUser.is_silenced] : [osu.trans("beatmap_discussions.reply.open.guest"), "fas fa-sign-in-alt", !1])[0], t = s[1], e = s[2], Object(m.div)({className: n + " " + n + "--reply " + n + "--new-reply " + n + "--new-reply-placeholder"}, i(a.a, {
                        disabled: e,
                        icon: t,
                        modifiers: "beatmap-discussion-reply-open",
                        props: {onClick: this.editStart},
                        text: r
                    }))
                }, d.prototype.renderReplyButton = function (e) {
                    return Object(m.div)({className: n + "__action"}, i(a.a, {
                        disabled: !this.validPost() || null != this.state.posting,
                        icon: s[e],
                        isBusy: this.state.posting === e,
                        text: osu.trans("common.buttons." + e),
                        props: {"data-action": e, onClick: this.throttledPost}
                    }))
                }, d.prototype.canReopen = function () {
                    return this.props.discussion.can_be_resolved && this.props.discussion.current_user_attributes.can_reopen
                }, d.prototype.canResolve = function () {
                    return this.props.discussion.can_be_resolved && this.props.discussion.current_user_attributes.can_resolve
                }, d.prototype.editStart = function () {
                    var e;
                    if (!c.a.userLogin.showIfGuest(this.editStart)) return this.setState({editing: !0}, (e = this, function () {
                        var t;
                        return null != (t = e.box.current) ? t.focus() : void 0
                    }))
                }, d.prototype.handleKeyDownCallback = function (e, t) {
                    switch (e) {
                        case InputHandler.CANCEL:
                            return this.setState({editing: !1});
                        case InputHandler.SUBMIT:
                            return this.throttledPost(t)
                    }
                }, d.prototype.isTimeline = function () {
                    return null != this.props.discussion.timestamp
                }, d.prototype.onCancelClick = function () {
                    if ("" === this.state.message || confirm(osu.trans("common.confirmation_unsaved"))) return this.setState({
                        editing: !1,
                        message: ""
                    })
                }, d.prototype.post = function (e) {
                    var t, s, n, i, a;
                    if (this.validPost()) return Object(h.b)(), null != (s = this.postXhr) && s.abort(), t = null != (n = e.currentTarget.dataset.action) ? n : "reply", this.setState({posting: t}), i = function () {
                        switch (t) {
                            case"reply_resolve":
                                return !0;
                            case"reply_reopen":
                                return !1;
                            default:
                                return null
                        }
                    }(), this.postXhr = r.ajax(Object(l.a)("beatmapsets.discussions.posts.store"), {
                        method: "POST",
                        data: {
                            beatmap_discussion_id: this.props.discussion.id,
                            beatmap_discussion: null != i ? {resolved: i} : {},
                            beatmap_discussion_post: {message: this.state.message}
                        }
                    }).done((a = this, function (e) {
                        return a.setState({
                            message: "",
                            editing: !1
                        }), r.publish("beatmapDiscussionPost:markRead", {id: e.beatmap_discussion_post_ids}), r.publish("beatmapsetDiscussions:update", {beatmapset: e.beatmapset})
                    })).fail(osu.ajaxError).always(function (e) {
                        return function () {
                            return Object(h.a)(), e.setState({posting: null})
                        }
                    }(this))
                }, d.prototype.setMessage = function (e) {
                    return this.setState({message: e.target.value})
                }, d.prototype.storageKey = function () {
                    return "beatmapset-discussion:reply:" + this.props.discussion.id + ":message"
                }, d.prototype.storeMessage = function () {
                    return "" === this.state.message ? localStorage.removeItem(this.storageKey()) : localStorage.setItem(this.storageKey(), this.state.message)
                }, d.prototype.storedMessage = function () {
                    var e;
                    return null != (e = localStorage.getItem(this.storageKey())) ? e : ""
                }, d.prototype.validPost = function () {
                    return BeatmapDiscussionHelper.validMessageLength(this.state.message, this.isTimeline())
                }, d
            }(u.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, vMSe: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return g
            }));
            var n, i, a = s("cQQh"), o = s("/HbY"), l = s("c1EF"), c = s("0h6b"), u = s("/G9H"), d = s("LQV2"),
                p = s.n(d), m = s("I8Ok"), h = s("/jJF"), b = s("tX/w"), f = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, v = {}.hasOwnProperty;
            i = u.createElement, n = "comment-editor";
            var g = function (t) {
                function s(t) {
                    var r;
                    this.post = f(this.post, this), this.onChange = f(this.onChange, this), this.mode = f(this.mode, this), this.isValid = f(this.isValid, this), this.handleKeyDownCallback = f(this.handleKeyDownCallback, this), this.close = f(this.close, this), this.buttonText = f(this.buttonText, this), this.render = f(this.render, this), this.componentWillUnmount = f(this.componentWillUnmount, this), this.componentDidMount = f(this.componentDidMount, this), s.__super__.constructor.call(this, t), this.textarea = u.createRef(), this.throttledPost = e.throttle(this.post, 1e3), this.handleKeyDown = InputHandler.textarea(this.handleKeyDownCallback), this.state = {
                        message: null != (r = this.props.message) ? r : "",
                        posting: !1
                    }
                }

                return function (e, t) {
                    for (var s in t) v.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.componentDidMount = function () {
                    var e, t, s;
                    if (null == (e = this.props.focus) || e) return null != (t = this.textarea.current) && (t.selectionStart = -1), null != (s = this.textarea.current) ? s.focus() : void 0
                }, s.prototype.componentWillUnmount = function () {
                    var e;
                    return this.throttledPost.cancel(), null != (e = this.xhr) ? e.abort() : void 0
                }, s.prototype.render = function () {
                    var e;
                    return e = Object(b.a)(n, this.props.modifiers), "new" === this.mode() && (e += " " + n + "--fancy"), Object(m.div)({className: e}, "new" === this.mode() ? Object(m.div)({className: n + "__avatar"}, i(l.a, {
                        user: currentUser,
                        modifiers: ["full-circle"]
                    })) : void 0, i(p.a, {
                        className: n + "__message",
                        ref: this.textarea,
                        value: this.state.message,
                        placeholder: osu.trans("comments.placeholder." + this.mode()),
                        onChange: this.onChange,
                        onKeyDown: this.handleKeyDown,
                        disabled: null == currentUser.id || this.state.posting
                    }), Object(m.div)({className: n + "__footer"}, Object(m.div)({className: n + "__footer-item " + n + "__footer-item--notice hidden-xs"}, osu.trans("comments.editor.textarea_hint._", {action: osu.trans("comments.editor.textarea_hint." + this.mode())})), null != this.props.close ? Object(m.div)({className: n + "__footer-item"}, i(a.a, {
                        disabled: this.state.posting,
                        modifiers: "comment-editor",
                        props: {onClick: this.props.close},
                        text: osu.trans("common.buttons.cancel")
                    })) : void 0, null != currentUser.id ? Object(m.div)({className: n + "__footer-item"}, i(a.a, {
                        disabled: this.state.posting || !this.isValid(),
                        isBusy: this.state.posting,
                        modifiers: "comment-editor",
                        props: {onClick: this.throttledPost},
                        text: {top: this.state.posting ? i(o.a, {modifiers: "center-inline"}) : this.buttonText()}
                    })) : Object(m.div)({className: n + "__footer-item"}, i(a.a, {
                        extraClasses: ["js-user-link"],
                        modifiers: "comment-editor",
                        text: osu.trans("comments.guest_button." + this.mode())
                    }))))
                }, s.prototype.buttonText = function () {
                    var e;
                    return e = function () {
                        switch (this.mode()) {
                            case"reply":
                                return "reply";
                            case"edit":
                                return "save";
                            case"new":
                                return "post"
                        }
                    }.call(this), osu.trans("common.buttons." + e)
                }, s.prototype.close = function () {
                    var e;
                    if (null != this.props.close && ((null != (e = this.props.message) ? e : "") === this.state.message || confirm(osu.trans("common.confirmation_unsaved")))) return this.props.close()
                }, s.prototype.handleKeyDownCallback = function (e, t) {
                    switch (e) {
                        case InputHandler.CANCEL:
                            return this.close();
                        case InputHandler.SUBMIT:
                            return this.throttledPost()
                    }
                }, s.prototype.isValid = function () {
                    return null != this.state.message && this.state.message.length > 0
                }, s.prototype.mode = function () {
                    return null != this.props.parent ? "reply" : null != this.props.id ? "edit" : "new"
                }, s.prototype.onChange = function (e) {
                    return this.setState({message: e.target.value})
                }, s.prototype.post = function () {
                    var e, t, s, n, i, a, o;
                    if ("edit" === this.mode() && this.state.message === this.props.message) return "function" == typeof (e = this.props).close ? e.close() : void 0;
                    switch (this.setState({posting: !0}), t = {comment: {message: this.state.message}}, this.mode()) {
                        case"reply":
                        case"new":
                            a = Object(c.a)("comments.store"), s = "POST", t.comment.commentable_type = this.props.commentableType, t.comment.commentable_id = this.props.commentableId, t.comment.parent_id = null != (i = this.props.parent) ? i.id : void 0, o = this, n = function (e) {
                                return o.setState({message: ""}), r.publish("comments:new", e)
                            };
                            break;
                        case"edit":
                            a = Object(c.a)("comments.update", {comment: this.props.id}), s = "PUT", n = function (e) {
                                return r.publish("comment:updated", e)
                            }
                    }
                    return this.xhr = r.ajax(a, {method: s, data: t}).always(function (e) {
                        return function () {
                            return e.setState({posting: !1})
                        }
                    }(this)).done(function (e) {
                        return function (t) {
                            var s, r;
                            return n(t), "function" == typeof (s = e.props).onPosted && s.onPosted(e.mode()), "function" == typeof (r = e.props).close ? r.close() : void 0
                        }
                    }(this)).fail(Object(h.c)(this.post))
                }, s
            }(u.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, vZz4: function (e, t, s) {
        "use strict";
        s.d(t, "j", (function () {
            return c
        })), s.d(t, "a", (function () {
            return u
        })), s.d(t, "b", (function () {
            return d
        })), s.d(t, "c", (function () {
            return p
        })), s.d(t, "d", (function () {
            return m
        })), s.d(t, "e", (function () {
            return h
        })), s.d(t, "h", (function () {
            return b
        })), s.d(t, "f", (function () {
            return f
        })), s.d(t, "g", (function () {
            return v
        })), s.d(t, "i", (function () {
            return g
        })), s.d(t, "k", (function () {
            return y
        }));
        var r = s("0h6b"), n = s("Hs9Z"), i = s("is6n"), a = function (e, t) {
            var s = {};
            for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (s[r] = e[r]);
            if (null != e && "function" == typeof Object.getOwnPropertySymbols) {
                var n = 0;
                for (r = Object.getOwnPropertySymbols(e); n < r.length; n++) t.indexOf(r[n]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[n]) && (s[r[n]] = e[r[n]])
            }
            return s
        };
        const o = ["admin", "api/v2", "beatmaps", "beatmapsets", "client-verifications", "comments", "community", "help", "home", "groups", "legal", "multiplayer", "notifications", "oauth", "rankings", "scores", "session", "store", "users", "wiki"].join("|"),
            l = RegExp(`^/(?:${o})(?:$|/|#)`),
            c = /(https?:\/\/((?:(?:[a-z0-9]\.|[a-z0-9][a-z0-9-]*[a-z0-9]\.)*[a-z][a-z0-9-]*[a-z0-9](?::\d+)?)(?:(?:(?:\/+(?:[a-z0-9$_.+!*',;:@&=-]|%[0-9a-f]{2})*)*(?:\?(?:[a-z0-9$_.+!*',;:@&=-]|%[0-9a-f]{2})*)?)?(?:#(?:[a-z0-9$_.+!*',;:@&=/?-]|%[0-9a-f]{2})*)?)?(?:[^.,:\s])))/gi;

        function u(e) {
            return `osu://b/${e}`
        }

        function d(e) {
            return `osu://s/${e}`
        }

        function p(e) {
            return Object(r.a)("changelog.build", {build: e.version, stream: e.update_stream.name})
        }

        function m(e) {
            return e.isHTML() || Object(n.startsWith)(e.getPath(), "/home/changelog/")
        }

        function h(e) {
            return l.test(e.getPath())
        }

        function b(e) {
            return `osu://edit/${e}`
        }

        function f(e, t, s) {
            (null == s ? void 0 : s.unescape) && (e = unescape(e), t = unescape(t));
            const r = document.createElement("a");
            if (r.textContent = t, r.setAttribute("href", e), null != s && (s.isRemote && r.setAttribute("data-remote", "1"), null != s.classNames && (r.className = s.classNames.join(" ")), null != s.props)) {
                const e = s.props, {className: t} = e, n = a(e, ["className"]);
                null != t && (r.className = t);
                for (const [s, i] of Object.entries(n)) null != i && r.setAttribute(s, i)
            }
            return r.outerHTML
        }

        function v(e, t = !1) {
            return e.replace(c, `<a href="$1" rel="nofollow noreferrer"${t ? ' target="_blank"' : ""}>$2</a>`)
        }

        function g(e, t) {
            const s = Object(i.a)(), r = new URL(null != e ? e : s.href, s.origin);
            for (const [n, i] of Object.entries(t)) null != i ? r.searchParams.set(n, i) : r.searchParams.delete(n);
            return r.href
        }

        function y(e, t) {
            return Object(r.a)("wiki.show", {
                locale: null != t ? t : currentLocale,
                path: "WIKI_PATH"
            }).replace("WIKI_PATH", encodeURI(e))
        }
    }, va1x: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return c
            }));
            var r = s("bpgk"), n = s("WLnA"), i = s("UZmH"), a = s("0h6b"), o = s("lv9K"), l = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };
            let c = class {
                constructor(e) {
                    Object.defineProperty(this, "beatmapsetStore", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e
                    }), Object.defineProperty(this, "recommendedDifficulties", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "resultSets", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new Map
                    }), Object.defineProperty(this, "xhr", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object(o.p)(this)
                }

                cancel() {
                    this.xhr && this.xhr.abort()
                }

                get(e, t = 0) {
                    if (t < 0) throw Error("from must be > 0");
                    const s = e.toKeyString(), r = this.getOrCreate(s);
                    return t > 0 && t < r.beatmapsetIds.size || 0 === t && !r.isExpired ? Promise.resolve(r) : this.fetch(e, t).then(s => (null != s && Object(o.u)(() => {
                        0 === t && r.reset(), this.updateBeatmapsetStore(s), r.append(s), this.recommendedDifficulties.set(e.mode, s.recommended_difficulty)
                    }), r))
                }

                getResultSet(e) {
                    const t = e.toKeyString();
                    return this.getOrCreate(t)
                }

                handleDispatchAction(e) {
                    e instanceof r.a && this.clear()
                }

                initialize(e, t) {
                    this.updateBeatmapsetStore(t);
                    const s = e.toKeyString(), r = this.getOrCreate(s);
                    null == r.fetchedAt && (r.append(t), this.recommendedDifficulties.set(e.mode, t.recommended_difficulty))
                }

                clear() {
                    this.resultSets.clear(), this.recommendedDifficulties.clear()
                }

                fetch(t, s) {
                    this.cancel();
                    const r = t.queryParams, n = t.toKeyString(), i = this.getOrCreate(n).cursorString;
                    if (s > 0) if (null != i) r.cursor_string = i; else if (null === i) return Promise.resolve(null);
                    const o = Object(a.a)("beatmapsets.search");
                    return this.xhr = e.ajax(o, {data: r, dataType: "json", method: "get"}), this.xhr
                }

                getOrCreate(e) {
                    let t = this.resultSets.get(e);
                    return null == t && (t = new i.a, this.resultSets.set(e, t)), t
                }

                updateBeatmapsetStore(e) {
                    for (const t of e.beatmapsets) this.beatmapsetStore.update(t)
                }
            };
            l([o.q], c.prototype, "recommendedDifficulties", void 0), l([o.q], c.prototype, "resultSets", void 0), l([o.f], c.prototype, "get", null), l([o.f], c.prototype, "initialize", null), l([o.f], c.prototype, "clear", null), c = l([n.b], c)
        }).call(this, s("5wds"))
    }, vp2U: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return n
        }));
        var r = s("/G9H");

        function n(e) {
            return null == e.score.best_id ? r.createElement("span", {title: osu.trans("scores.status.non_best")}, "-") : null == e.score.pp ? r.createElement("span", {
                className: "fas fa-exclamation-triangle",
                title: osu.trans("scores.status.processing")
            }) : r.createElement("span", {title: osu.formatNumber(e.score.pp)}, osu.formatNumber(Math.round(e.score.pp)), e.suffix)
        }
    }, w1mS: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return a
        }));
        var r = s("/G9H"), n = s("tX/w");
        const i = "value-display";

        function a({description: e, label: t, modifiers: s, value: a}) {
            return r.createElement("div", {className: Object(n.a)(i, s)}, r.createElement("div", {className: `${i}__label`}, t), r.createElement("div", {className: `${i}__value`}, a), null != e && r.createElement("div", {className: `${i}__description`}, e))
        }
    }, wsBb: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return o
        }));
        var r = s("f4vq"), n = s("/G9H"), i = s("tX/w"), a = s("elBb");

        class o extends n.Component {
            render() {
                var e, t;
                const s = Object(i.a)("header-v4", osu.presence(this.props.theme), {restricted: null !== (t = null === (e = r.a.currentUser) || void 0 === e ? void 0 : e.is_restricted) && void 0 !== t && t}, this.props.modifiers);
                return n.createElement("div", {className: s}, n.createElement("div", {className: "header-v4__container header-v4__container--main"}, n.createElement("div", {className: "header-v4__bg-container"}, n.createElement("div", {
                    className: "header-v4__bg",
                    style: {backgroundImage: osu.urlPresence(this.props.backgroundImage)}
                })), n.createElement("div", {className: "header-v4__content"}, this.props.contentPrepend, n.createElement("div", {className: "header-v4__row header-v4__row--title"}, n.createElement("div", {className: "header-v4__icon"}), n.createElement("div", {className: "header-v4__title"}, this.title()), this.props.titleAppend), this.props.contentAppend)), this.props.links.length > 0 && n.createElement("div", {className: "header-v4__container"}, n.createElement("div", {className: "header-v4__content"}, n.createElement("div", {className: "header-v4__row header-v4__row--bar"}, this.renderLinks(), this.renderLinksMobile()))))
            }

            renderLinks() {
                const e = this.props.links.map(e => {
                    const t = [];
                    return e.active && t.push("active"), n.createElement("li", {
                        key: `${e.url}-${e.title}`,
                        className: "header-nav-v4__item"
                    }, n.createElement("a", Object.assign({
                        className: Object(i.a)("header-nav-v4__link", t),
                        href: e.url,
                        onClick: this.props.onLinkClick
                    }, e.data), n.createElement("span", {className: "fake-bold", "data-content": e.title}, e.title)))
                }), t = this.props.linksBreadcrumb ? "ol" : "ul", s = [];
                return s.push(this.props.linksBreadcrumb ? "breadcrumb" : "list"), n.createElement(t, {className: Object(i.a)("header-nav-v4", s)}, e)
            }

            renderLinksMobile() {
                if (this.props.linksBreadcrumb) return null;
                if (0 === this.props.links.length) return null;
                let e = this.props.links[0];
                const t = this.props.links.map(t => {
                    const s = [];
                    return t.active && (s.push("active"), e = t), n.createElement("li", {key: `${t.url}-${t.title}`}, n.createElement("a", Object.assign({
                        className: "header-nav-mobile__item js-click-menu--close",
                        href: t.url,
                        onClick: this.props.onLinkClick
                    }, t.data), t.title))
                });
                return n.createElement("div", {className: "header-nav-mobile"}, n.createElement("a", {
                    className: "header-nav-mobile__toggle js-click-menu",
                    "data-click-menu-target": "header-nav-mobile",
                    href: e.url
                }, e.title, n.createElement("span", {className: "header-nav-mobile__toggle-icon"}, n.createElement("span", {className: "fas fa-chevron-down"}))), n.createElement("ul", {
                    className: "header-nav-mobile__menu js-click-menu",
                    "data-click-menu-id": "header-nav-mobile",
                    "data-visibility": "hidden"
                }, t))
            }

            title() {
                const e = Object(a.b)("json-route-section"),
                    t = [`page_title.${e.namespace}.${e.controller}.${e.action}`, `page_title.${e.namespace}.${e.controller}._`, `page_title.${e.namespace}._`];
                for (const s of t) if (osu.transExists(s, fallbackLocale)) return osu.trans(s);
                return "unknown"
            }
        }

        Object.defineProperty(o, "defaultProps", {
            enumerable: !0,
            configurable: !0,
            writable: !0,
            value: {links: [], linksBreadcrumb: !1}
        })
    }, x6t3: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return i
            }));
            var r = s("lv9K"), n = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

            class i {
                constructor() {
                    Object.defineProperty(this, "privateIsDesktop", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "handleResize", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: () => {
                            this.privateIsDesktop = window.matchMedia("(min-width: 900px)").matches;
                            const e = .01 * window.innerHeight;
                            document.documentElement.style.setProperty("--vh", `${e}px`)
                        }
                    }), this.handleResize(), Object(r.p)(this), e(window).on("resize", this.handleResize)
                }

                get isDesktop() {
                    return this.privateIsDesktop
                }

                get isMobile() {
                    return !this.privateIsDesktop
                }
            }

            n([r.q], i.prototype, "privateIsDesktop", void 0), n([r.f], i.prototype, "handleResize", void 0)
        }).call(this, s("5wds"))
    }, xO1E: function (e, t, s) {
        "use strict";
        var r = s("PdfH"), n = s("VbpL"), i = s("/G9H");

        function a({level: e}) {
            return i.createElement("div", {
                className: "user-level",
                title: osu.trans("users.show.stats.level", {level: e})
            }, e)
        }

        var o = s("0h6b"), l = s("KUml"), c = s("f4vq"), u = s("7nlf"), d = s("AqrC"), p = s("B5xN");

        function m(e) {
            return i.createElement("div", {
                className: "btn-circle btn-circle--page-toggle btn-circle--page-toggle-detail",
                title: osu.trans("common.buttons.show_more_options")
            }, i.createElement(d.a, null, t => i.createElement("div", {className: "simple-menu"}, i.createElement(u.a, {
                modifiers: "inline",
                onClick: t,
                userId: e.user.id,
                wrapperClass: "simple-menu__item"
            }), i.createElement(p.a, {
                className: "simple-menu__item",
                icon: !0,
                onFormClose: t,
                reportableId: e.user.id.toString(),
                reportableType: "user",
                user: e.user
            }))))
        }

        let h = class extends i.Component {
            get showMessageButton() {
                return null == c.a.currentUser || c.a.currentUser.id !== this.props.user.id && !c.a.currentUserModel.blocks.has(this.props.user.id)
            }

            render() {
                return i.createElement("div", {className: "profile-detail-bar"}, i.createElement(n.a, {
                    alwaysVisible: !0,
                    followers: this.props.user.follower_count,
                    modifiers: "profile-page",
                    userId: this.props.user.id
                }), this.renderNonBotButtons())
            }

            renderNonBotButtons() {
                return this.props.user.is_bot ? null : i.createElement(i.Fragment, null, i.createElement(r.a, {
                    alwaysVisible: !0,
                    followers: this.props.user.mapping_follower_count,
                    modifiers: "profile-page",
                    showFollowerCounter: !0,
                    userId: this.props.user.id
                }), this.showMessageButton && i.createElement("div", null, i.createElement("a", {
                    className: "user-action-button user-action-button--profile-page",
                    href: Object(o.a)("messages.users.show", {user: this.props.user.id}),
                    title: osu.trans("users.card.send_message")
                }, i.createElement("i", {className: "fas fa-envelope"}))), (e = this.props.user, null != c.a.currentUser && c.a.currentUser.id !== e.id && i.createElement(m, {user: this.props.user})), null != this.props.user.statistics && i.createElement("div", {className: "profile-detail-bar__level"}, i.createElement("div", {className: "profile-detail-bar__level-bar"}, i.createElement("div", {
                    className: "bar bar--user-profile",
                    title: osu.trans("users.show.stats.level_progress")
                }, i.createElement("div", {
                    className: "bar__fill",
                    style: {width: `${this.props.user.statistics.level.progress}%`}
                }), i.createElement("div", {className: "bar__text"}, `${this.props.user.statistics.level.progress}%`))), i.createElement(a, {level: this.props.user.statistics.level.current})));
                var e
            }
        };
        h = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([l.b], h);
        t.a = h
    }, y1is: function (e, t, s) {
        "use strict";
        (function (e, r) {
            s.d(t, "a", (function () {
                return S
            }));
            var n, i, a = s("Y9Pv"), o = s("DHbW"), l = s("ufn5"), c = s("XMXc"), u = s("cQQh"), d = s("tGwB"),
                p = s("B5xN"), m = s("55pz"), h = s("UBw1"), b = s("yHuj"), f = s("0h6b"), v = s("srn7"), g = s("/G9H"),
                y = s("LQV2"), w = s.n(y), _ = s("I8Ok"), O = s("iH//"), j = s("tX/w"), E = s("pKj0"), P = s("C3HX"),
                k = function (e, t) {
                    return function () {
                        return e.apply(t, arguments)
                    }
                }, N = {}.hasOwnProperty;
            i = g.createElement, n = "beatmap-discussion-post";
            var S = function (t) {
                function s(t) {
                    this.validPost = k(this.validPost, this), this.updatePost = k(this.updatePost, this), this.updateCanSave = k(this.updateCanSave, this), this.setMessage = k(this.setMessage, this), this.isTimeline = k(this.isTimeline, this), this.isReview = k(this.isReview, this), this.canReport = k(this.canReport, this), this.messageViewer = k(this.messageViewer, this), this.messageEditor = k(this.messageEditor, this), this.isOwner = k(this.isOwner, this), this.handleKeyDownCallback = k(this.handleKeyDownCallback, this), this.editStart = k(this.editStart, this), this.editCancel = k(this.editCancel, this), this.render = k(this.render, this), this.componentWillUnmount = k(this.componentWillUnmount, this), s.__super__.constructor.call(this, t), this.textareaRef = g.createRef(), this.messageBodyRef = g.createRef(), this.throttledUpdatePost = e.throttle(this.updatePost, 1e3), this.handleKeyDown = InputHandler.textarea(this.handleKeyDownCallback), this.xhr = {}, this.reviewEditor = g.createRef(), this.state = {
                        canSave: !0,
                        editing: !1,
                        editorMinHeight: "0",
                        posting: !1,
                        message: null
                    }
                }

                return function (e, t) {
                    for (var s in t) N.call(t, s) && (e[s] = t[s]);

                    function r() {
                        this.constructor = e
                    }

                    r.prototype = t.prototype, e.prototype = new r, e.__super__ = t.prototype
                }(s, t), s.prototype.componentWillUnmount = function () {
                    var e, t, s, r;
                    for (e in this.throttledUpdatePost.cancel(), s = [], t = this.xhr) N.call(t, e) && (r = t[e], s.push(null != r ? r.abort() : void 0));
                    return s
                }, s.prototype.render = function () {
                    var e, t, s;
                    return t = Object(j.a)(n, ((e = {})["" + this.props.type] = !0, e.deleted = null != this.props.post.deleted_at, e.editing = this.state.editing, e.unread = !this.props.read && "discussion" !== this.props.type, e)), t += " js-beatmap-discussion-jump", Object(_.div)({
                        className: t,
                        "data-post-id": this.props.post.id,
                        key: this.props.type + "-" + this.props.post.id,
                        onClick: (s = this, function () {
                            return r.publish("beatmapDiscussionPost:markRead", {id: s.props.post.id})
                        })
                    }, Object(_.div)({className: n + "__content"}, "reply" === this.props.type ? i(P.a, {
                        user: this.props.user,
                        group: Object(O.a)({
                            beatmapset: this.props.beatmapset,
                            currentBeatmap: this.props.beatmap,
                            discussion: this.props.discussion,
                            user: this.props.user
                        })
                    }) : void 0, this.state.editing ? this.messageEditor() : this.messageViewer()))
                }, s.prototype.editCancel = function () {
                    return this.setState({editing: !1})
                }, s.prototype.editStart = function () {
                    var e, t;
                    return null != this.messageBodyRef.current && (e = this.messageBodyRef.current.getBoundingClientRect().height + 50 + "px"), this.setState({
                        editing: !0,
                        editorMinHeight: null != e ? e : "0",
                        message: this.props.post.message
                    }, (t = this, function () {
                        var e;
                        return null != (e = t.textareaRef.current) ? e.focus() : void 0
                    }))
                }, s.prototype.handleKeyDownCallback = function (e, t) {
                    switch (e) {
                        case InputHandler.SUBMIT:
                            return this.throttledUpdatePost()
                    }
                }, s.prototype.isOwner = function () {
                    return this.props.post.user_id === this.props.beatmapset.user_id
                }, s.prototype.messageEditor = function () {
                    var e, t;
                    if (this.props.canBeEdited) return e = !this.state.posting && this.state.canSave, Object(_.div)({className: n + "__message-container"}, this.isReview() ? i(o.a.Consumer, null, (t = this, function (e) {
                        return i(a.a.Consumer, null, (function (s) {
                            return i(l.a, {
                                beatmapset: t.props.beatmapset,
                                beatmaps: s,
                                document: t.props.post.message,
                                discussion: t.props.discussion,
                                discussions: e,
                                editMode: !0,
                                editing: t.state.editing,
                                ref: t.reviewEditor,
                                onChange: t.updateCanSave
                            })
                        }))
                    })) : i(g.Fragment, null, i(w.a, {
                        style: {minHeight: this.state.editorMinHeight},
                        disabled: this.state.posting,
                        className: n + "__message " + n + "__message--editor",
                        onChange: this.setMessage,
                        onKeyDown: this.handleKeyDown,
                        value: this.state.message,
                        ref: this.textareaRef
                    }), i(E.a, {
                        message: this.state.message,
                        isTimeline: this.isTimeline()
                    })), Object(_.div)({className: n + "__actions"}, Object(_.div)({className: n + "__actions-group"}), Object(_.div)({className: n + "__actions-group"}, Object(_.div)({className: n + "__action"}, i(u.a, {
                        disabled: this.state.posting,
                        props: {onClick: this.editCancel},
                        text: osu.trans("common.buttons.cancel")
                    })), Object(_.div)({className: n + "__action"}, i(u.a, {
                        disabled: !e,
                        props: {onClick: this.throttledUpdatePost},
                        text: osu.trans("common.buttons.save")
                    })))))
                }, s.prototype.messageViewer = function () {
                    var e, t, s, r, a, o, l, u;
                    return e = (o = "reply" === this.props.type ? ["beatmapsets.discussions.posts", "post", this.props.post] : ["beatmapsets.discussions", "discussion", this.props.discussion])[0], s = o[1], t = o[2], Object(_.div)({className: n + "__message-container"}, this.isReview() ? Object(_.div)({className: n + "__message"}, i(c.a, {
                        discussions: this.context.discussions,
                        message: this.props.post.message
                    })) : Object(_.div)({
                        className: n + "__message",
                        ref: this.messageBodyRef,
                        dangerouslySetInnerHTML: {__html: BeatmapDiscussionHelper.format(this.props.post.message)}
                    }), Object(_.div)({className: n + "__info-container"}, Object(_.span)({className: n + "__info"}, i(h.a, {
                        dateTime: this.props.post.created_at,
                        relative: !0
                    })), null != t.deleted_at ? Object(_.span)({className: n + "__info " + n + "__info--edited"}, i(m.a, {
                        mappings: {
                            editor: i(b.a, {
                                className: n + "__info-user",
                                user: null != (l = this.props.users[t.deleted_by_id]) ? l : v.b
                            }), delete_time: i(h.a, {dateTime: t.deleted_at, relative: !0})
                        }, pattern: osu.trans("beatmaps.discussions.deleted")
                    })) : void 0, this.props.post.updated_at !== this.props.post.created_at && null != this.props.lastEditor ? Object(_.span)({className: n + "__info " + n + "__info--edited"}, i(m.a, {
                        mappings: {
                            editor: i(b.a, {
                                className: n + "__info-user",
                                user: this.props.lastEditor
                            }), update_time: i(h.a, {dateTime: this.props.post.updated_at, relative: !0})
                        }, pattern: osu.trans("beatmaps.discussions.edited")
                    })) : void 0, "discussion" === this.props.type && this.props.discussion.kudosu_denied ? Object(_.span)({className: n + "__info " + n + "__info--edited"}, osu.trans("beatmaps.discussions.kudosu_denied")) : void 0), Object(_.div)({className: n + "__actions"}, Object(_.div)({className: n + "__actions-group"}, Object(_.span)({className: n + "__action " + n + "__action--button"}, i(d.a, {
                        value: BeatmapDiscussionHelper.url({
                            discussion: this.props.discussion,
                            post: "reply" === this.props.type ? this.props.post : void 0
                        }), label: osu.trans("common.buttons.permalink"), valueAsUrl: !0
                    })), this.props.canBeEdited ? Object(_.button)({
                        className: n + "__action " + n + "__action--button",
                        onClick: this.editStart
                    }, osu.trans("beatmaps.discussions.edit")) : void 0, null == t.deleted_at && this.props.canBeDeleted ? Object(_.a)({
                        className: "js-beatmapset-discussion-update " + n + "__action " + n + "__action--button",
                        href: Object(f.a)(e + ".destroy", (r = {}, r["" + s] = t.id, r)),
                        "data-remote": !0,
                        "data-method": "DELETE",
                        "data-confirm": osu.trans("common.confirmation")
                    }, osu.trans("beatmaps.discussions.delete")) : void 0, null != t.deleted_at && this.props.canBeRestored ? Object(_.a)({
                        className: "js-beatmapset-discussion-update " + n + "__action " + n + "__action--button",
                        href: Object(f.a)(e + ".restore", (a = {}, a["" + s] = t.id, a)),
                        "data-remote": !0,
                        "data-method": "POST",
                        "data-confirm": osu.trans("common.confirmation")
                    }, osu.trans("beatmaps.discussions.restore")) : void 0, "discussion" === this.props.type && (null != (u = this.props.discussion.current_user_attributes) ? u.can_moderate_kudosu : void 0) ? this.props.discussion.can_grant_kudosu ? Object(_.a)({
                        className: "js-beatmapset-discussion-update " + n + "__action " + n + "__action--button",
                        href: Object(f.a)("beatmapsets.discussions.deny-kudosu", {discussion: this.props.discussion.id}),
                        "data-remote": !0,
                        "data-method": "POST",
                        "data-confirm": osu.trans("common.confirmation")
                    }, osu.trans("beatmaps.discussions.deny_kudosu")) : this.props.discussion.kudosu_denied ? Object(_.a)({
                        className: "js-beatmapset-discussion-update " + n + "__action " + n + "__action--button",
                        href: Object(f.a)("beatmapsets.discussions.allow-kudosu", {discussion: this.props.discussion.id}),
                        "data-remote": !0,
                        "data-method": "POST",
                        "data-confirm": osu.trans("common.confirmation")
                    }, osu.trans("beatmaps.discussions.allow_kudosu")) : void 0 : void 0, this.canReport() ? i(p.a, {
                        className: n + "__action " + n + "__action--button",
                        reportableId: this.props.post.id,
                        reportableType: "beatmapset_discussion_post",
                        user: this.props.user
                    }) : void 0)))
                }, s.prototype.canReport = function () {
                    return null != currentUser.id && this.props.post.user_id !== currentUser.id
                }, s.prototype.isReview = function () {
                    return "review" === this.props.discussion.message_type && "discussion" === this.props.type
                }, s.prototype.isTimeline = function () {
                    return null != this.props.discussion.timestamp
                }, s.prototype.setMessage = function (e) {
                    return this.setState({message: e.target.value}, this.updateCanSave)
                }, s.prototype.updateCanSave = function () {
                    return this.setState({canSave: this.validPost()})
                }, s.prototype.updatePost = function () {
                    var t, s, n;
                    if (t = this.state.message, this.isReview()) {
                        if (t = this.reviewEditor.current.serialize(), e.isEqual(JSON.parse(this.props.post.message), JSON.parse(t))) return void this.setState({editing: !1});
                        if (!this.reviewEditor.current.showConfirmationIfRequired()) return;
                        this.setState({message: t})
                    }
                    if (t !== this.props.post.message) return this.setState({posting: !0}), null != (s = this.xhr.updatePost) && s.abort(), this.xhr.updatePost = r.ajax(Object(f.a)("beatmapsets.discussions.posts.update", {post: this.props.post.id}), {
                        method: "PUT",
                        data: {beatmap_discussion_post: {message: t}}
                    }).done((n = this, function (e) {
                        return n.setState({editing: !1}), r.publish("beatmapsetDiscussions:update", {beatmapset: e})
                    })).fail(osu.ajaxError).always(function (e) {
                        return function () {
                            return e.setState({posting: !1})
                        }
                    }(this));
                    this.setState({editing: !1})
                }, s.prototype.validPost = function () {
                    var e;
                    return this.isReview() ? null != (e = this.reviewEditor.current) ? e.canSave : void 0 : BeatmapDiscussionHelper.validMessageLength(this.state.message, this.isTimeline())
                }, s
            }(g.PureComponent)
        }).call(this, s("Hs9Z"), s("5wds"))
    }, y2EG: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return c
        }));
        var r = s("Hs9Z"), n = s("lv9K"), i = s("71br"), a = s("5evE");
        var o = s("f4vq"), l = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        };

        class c {
            constructor(e, t) {
                Object.defineProperty(this, "id", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e
                }), Object.defineProperty(this, "objectType", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: t
                }), Object.defineProperty(this, "createdAtJson", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "details", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: Object(i.a)()
                }), Object.defineProperty(this, "isDeleting", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isMarkingAsRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "isRead", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: !1
                }), Object.defineProperty(this, "name", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "objectId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "sourceUserId", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: void 0
                }), Object.defineProperty(this, "updateFromJson", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        var t, s;
                        return this.createdAtJson = e.created_at, this.isRead = e.is_read, this.name = e.name, this.objectId = e.object_id, this.sourceUserId = e.source_user_id, this.details = Object(i.a)(), "object" == typeof e.details && (Object(r.forEach)(e.details, (e, t) => {
                            this.details[Object(r.camelCase)(t)] = e
                        }), "comment_new" === e.name && (null === (t = e.details.reply_to) || void 0 === t ? void 0 : t.user_id) === (null === (s = o.a.currentUser) || void 0 === s ? void 0 : s.id) && (this.name = "comment_reply")), this
                    }
                }), Object(n.p)(this)
            }

            get canMarkRead() {
                return this.id > 0 && !this.isRead
            }

            get category() {
                var e;
                return Object(a.a)(null !== (e = this.name) && void 0 !== e ? e : "")
            }

            get categoryGroupKey() {
                return Object(a.b)(this)
            }

            get displayType() {
                return "legacy_pm" === (e = this).name ? "legacy_pm" : null != e.objectType && null != e.objectId ? "user_achievement_unlock" === e.name ? "user_achievement" : e.objectType : void 0;
                var e
            }

            get identity() {
                return {category: this.category, id: this.id, objectId: this.objectId, objectType: this.objectType}
            }

            get stackId() {
                return `${this.objectType}-${this.objectId}-${this.category}`
            }

            get title() {
                var e;
                return o.a.userPreferences.get("beatmapset_title_show_original") && null !== (e = osu.presence(this.details.titleUnicode)) && void 0 !== e ? e : this.details.title
            }

            static fromJson(e) {
                return new c(e.id, e.object_type).updateFromJson(e)
            }
        }

        l([n.q], c.prototype, "isDeleting", void 0), l([n.q], c.prototype, "isMarkingAsRead", void 0), l([n.q], c.prototype, "isRead", void 0), l([n.h], c.prototype, "canMarkRead", null), l([n.h], c.prototype, "category", null), l([n.h], c.prototype, "categoryGroupKey", null), l([n.h], c.prototype, "displayType", null), l([n.h], c.prototype, "stackId", null), l([n.h], c.prototype, "title", null)
    }, yHuj: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return i
        }));
        var r = s("0h6b"), n = s("/G9H");

        class i extends n.PureComponent {
            render() {
                var e;
                let t = "js-usercard";
                null != this.props.className && (t += ` ${this.props.className}`);
                const s = this.props.user.id ? Object(r.a)("users.show", {
                    mode: this.props.mode,
                    user: this.props.user.id
                }) : void 0;
                return n.createElement("a", {
                    className: t,
                    "data-tooltip-position": this.props.tooltipPosition,
                    "data-user-id": this.props.user.id,
                    href: s
                }, null !== (e = this.props.children) && void 0 !== e ? e : this.props.user.username)
            }
        }
    }, yJmy: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return c
        }));
        var r = s("UxfW"), n = s("/G9H"), i = s("tSlR"), a = s("dTpI"), o = s("TezV"), l = s("ss8h");

        class c extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "select", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        const t = "all" !== e ? parseInt(e, 10) : void 0,
                            s = a.b.findPath(this.context, this.props.element);
                        i.h.setNodes(this.context, {beatmapId: t}, {at: s})
                    }
                })
            }

            render() {
                var e;
                const t = [];
                return t.push({
                    icon: n.createElement("i", {className: "fas fa-fw fa-star-of-life"}),
                    id: "all",
                    label: osu.trans("beatmaps.discussions.mode.scopes.generalAll")
                }), this.props.beatmaps.forEach(e => {
                    e.deleted_at || t.push({
                        icon: n.createElement(r.a, {beatmap: e}),
                        id: e.id.toString(),
                        label: e.version
                    })
                }), n.createElement(o.a, {
                    disabled: this.props.disabled,
                    menuOptions: t,
                    onSelect: this.select,
                    selected: null === (e = this.props.element.beatmapId) || void 0 === e ? void 0 : e.toString()
                })
            }
        }

        Object.defineProperty(c, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: l.a})
    }, yoRQ: function (e, t, s) {
        "use strict";
        var r = s("KUml"), n = s("tnRH"), i = s("/G9H"), a = s("PVx+"), o = s("vZz4"), l = s("R9Sp"), c = s("55pz"),
            u = s("UBw1"), d = s("w1mS");

        function p({kudosu: e}) {
            const t = {
                amount: i.createElement("strong", {className: "profile-extra-entries__kudosu-amount"}, osu.trans("users.show.extra.kudosu.entry.amount", {amount: osu.formatNumber(Math.abs(e.amount))})),
                giver: null == e.giver ? osu.trans("users.deleted") : i.createElement("a", {href: e.giver.url}, e.giver.username),
                post: null == e.post.url ? e.post.title : i.createElement("a", {href: e.post.url}, e.post.title)
            };
            return i.createElement("li", {className: "profile-extra-entries__item"}, i.createElement("div", {className: "profile-extra-entries__detail"}, i.createElement("div", {className: "profile-extra-entries__text"}, i.createElement(c.a, {
                mappings: t,
                pattern: osu.trans(`users.show.extra.kudosu.entry.${e.model}.${e.action}`)
            }))), i.createElement("div", {className: "profile-extra-entries__time"}, i.createElement(u.a, {
                dateTime: e.created_at,
                relative: !0
            })))
        }

        let m = class extends i.Component {
            render() {
                return i.createElement("div", {className: "page-extra"}, i.createElement(n.a, {
                    name: this.props.name,
                    withEdit: this.props.withEdit
                }), i.createElement("div", {className: "kudosu-box"}, i.createElement(d.a, {
                    description: i.createElement(c.a, {
                        mappings: {link: i.createElement("a", {href: Object(o.k)("Kudosu")}, osu.trans("users.show.extra.kudosu.total_info.link"))},
                        pattern: osu.trans("users.show.extra.kudosu.total_info._")
                    }),
                    label: osu.trans("users.show.extra.kudosu.total"),
                    modifiers: "kudosu",
                    value: osu.formatNumber(this.props.total)
                })), this.renderEntries())
            }

            renderEntries() {
                return 0 === Object(a.d)(this.props.kudosu.items) ? i.createElement("div", {className: "profile-extra-entries profile-extra-entries--kudosu"}, osu.trans("users.show.extra.kudosu.entry.empty")) : i.createElement("ul", {className: "profile-extra-entries profile-extra-entries--kudosu"}, Array.isArray(this.props.kudosu.items) && this.props.kudosu.items.map(e => i.createElement(p, {
                    key: e.id,
                    kudosu: e
                })), i.createElement("li", {className: "profile-extra-entries__item"}, i.createElement(l.a, Object.assign({}, this.props.kudosu.pagination, {
                    callback: this.props.onShowMore,
                    modifiers: "profile-page"
                }))))
            }
        };
        m = function (e, t, s, r) {
            var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
            if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
            return i > 3 && a && Object.defineProperty(t, s, a), a
        }([r.b], m), t.a = m
    }, yun6: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "b", (function () {
                return u
            })), s.d(t, "a", (function () {
                return d
            })), s.d(t, "d", (function () {
                return p
            })), s.d(t, "c", (function () {
                return m
            })), s.d(t, "e", (function () {
                return h
            })), s.d(t, "f", (function () {
                return b
            })), s.d(t, "g", (function () {
                return f
            })), s.d(t, "h", (function () {
                return v
            }));
            var r = s("kUxX"), n = s("dMN7"), i = s("hoYT"), a = s("Hs9Z"), o = s("f4vq"), l = s("elBb");
            const c = r.scaleLinear().domain([.1, 1.25, 2, 2.5, 3.3, 4.2, 4.9, 5.8, 6.7, 7.7, 9]).clamp(!0).range(["#4290FB", "#4FC0FF", "#4FFFD5", "#7CFF4F", "#F6F05C", "#FF8068", "#FF4E6F", "#C645B8", "#6563DE", "#18158E", "#000000"]).interpolate(r.interpolateRgb.gamma(2.2));

            function u(t) {
                var s, r, o;
                if (null != t.items) {
                    let o, c = null;
                    const u = function (t) {
                        var s, r;
                        null == y && (y = null !== (s = Object(l.c)("json-recommended-star-difficulty-all")) && void 0 !== s ? s : {}, e(document).one("turbolinks:before-cache", () => {
                            y = null
                        }));
                        return null !== (r = y[t]) && void 0 !== r ? r : 1
                    }(null !== (s = t.mode) && void 0 !== s ? s : i.a[0]);
                    return t.items.forEach(e => {
                        const t = Math.abs(e.difficulty_rating - u);
                        (function (e) {
                            return !Object(n.a)(e) || null == e.deleted_at && !e.convert
                        })(e) && (null == o || t < o) && (o = t, c = e)
                    }), null !== (r = null != c ? c : a.last(t.items)) && void 0 !== r ? r : null
                }
                if (null == t.group) return null;
                const c = null == t.mode ? g() : [t.mode];
                for (const e of c) {
                    const s = u({items: null !== (o = t.group.get(e)) && void 0 !== o ? o : [], mode: e});
                    if (null != s) return s
                }
                return null
            }

            function d(e) {
                var t;
                const s = null == e.mode ? g() : [e.mode];
                for (const r of s) {
                    const s = null === (t = e.group.get(r)) || void 0 === t ? void 0 : t.find(t => t.id === e.id);
                    if (null != s) return s
                }
                return null
            }

            function p(e) {
                return e < .1 ? "#AAAAAA" : e >= 9 ? "#000000" : c(e)
            }

            function m(e) {
                return o.a.userPreferences.get("beatmapset_title_show_original") ? e.artist_unicode : e.artist
            }

            function h(e) {
                return o.a.userPreferences.get("beatmapset_title_show_original") ? e.title_unicode : e.title
            }

            function b(e) {
                const t = a.groupBy(null != e ? e : [], "mode"), s = new Map;
                return i.a.forEach(e => {
                    var r;
                    s.set(e, function (e) {
                        if (0 === e.length) return [];
                        if ("mania" === e[0].mode) return a.orderBy(e, ["convert", "cs", "difficulty_rating"], ["desc", "asc", "asc"]);
                        return a.orderBy(e, ["convert", "difficulty_rating"], ["desc", "asc"])
                    }(null !== (r = t[e]) && void 0 !== r ? r : []))
                }), s
            }

            function f(e) {
                return "ranked" === e.status || "approved" === e.status
            }

            function v(e) {
                return [...b(e).values()].flat()
            }

            function g() {
                var e;
                const t = null === (e = o.a.currentUser) || void 0 === e ? void 0 : e.playmode;
                if (null == t || !i.a.includes(t)) return i.a;
                const s = a.without(i.a, t);
                return s.unshift(t), s
            }

            let y = null
        }).call(this, s("5wds"))
    }, zFzD: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return p
            }));
            var r = s("UZmH"), n = s("WD1s"), i = s("0h6b"), a = s("Hs9Z"), o = s("lv9K"), l = s("f4vq"), c = s("is6n"),
                u = function (e, t, s, r) {
                    var n, i = arguments.length,
                        a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                    if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                    return i > 3 && a && Object.defineProperty(t, s, a), a
                }, d = function (e, t, s, r) {
                    return new (s || (s = Promise))((function (n, i) {
                        function a(e) {
                            try {
                                l(r.next(e))
                            } catch (t) {
                                i(t)
                            }
                        }

                        function o(e) {
                            try {
                                l(r.throw(e))
                            } catch (t) {
                                i(t)
                            }
                        }

                        function l(e) {
                            var t;
                            e.done ? n(e.value) : (t = e.value, t instanceof s ? t : new s((function (e) {
                                e(t)
                            }))).then(a, o)
                        }

                        l((r = r.apply(e, t || [])).next())
                    }))
                };

            class p {
                constructor(e) {
                    Object.defineProperty(this, "beatmapsetSearch", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e
                    }), Object.defineProperty(this, "advancedSearch", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "currentResultSet", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: new r.a
                    }), Object.defineProperty(this, "filters", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "isExpanded", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "searchStatus", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: {error: null, from: 0, state: "completed"}
                    }), Object.defineProperty(this, "debouncedFilterChangedSearch", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: Object(a.debounce)(this.filterChangedSearch, 500)
                    }), Object.defineProperty(this, "filtersObserver", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "initialErrorMessage", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "filterChangedHandler", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: e => {
                            "update" === e.type && e.oldValue === e.newValue || "remove" !== e.type && "sort" === e.name && null == e.newValue || (this.searchStatus.state = "input", this.debouncedFilterChangedSearch(), "query" !== e.name && this.debouncedFilterChangedSearch.flush())
                        }
                    }), Object(o.p)(this)
                }

                get currentBeatmapsetIds() {
                    return [...this.currentResultSet.beatmapsetIds]
                }

                get error() {
                    return this.searchStatus.error
                }

                get hasMore() {
                    return this.currentResultSet.hasMoreForPager
                }

                get isBusy() {
                    return "searching" === this.searchStatus.state || "input" === this.searchStatus.state
                }

                get isPaging() {
                    return "paging" === this.searchStatus.state
                }

                get isSupporterMissing() {
                    var e, t;
                    return !(null !== (t = null === (e = l.a.currentUser) || void 0 === e ? void 0 : e.is_supporter) && void 0 !== t && t) && BeatmapsetFilter.supporterRequired(this.filters).length > 0
                }

                get recommendedDifficulty() {
                    return this.beatmapsetSearch.recommendedDifficulties.get(this.filters.mode)
                }

                get supporterRequiredFilterText() {
                    const e = BeatmapsetFilter.supporterRequired(this.filters),
                        t = Object(a.map)(e, e => osu.trans(`beatmaps.listing.search.filters.${e}`));
                    return osu.transArray(t)
                }

                cancel() {
                    this.debouncedFilterChangedSearch.cancel(), this.beatmapsetSearch.cancel()
                }

                initialize(e) {
                    this.restoreStateFromUrl(), this.beatmapsetSearch.initialize(this.filters, e), this.initialErrorMessage = e.error
                }

                loadMore() {
                    !this.isBusy && this.hasMore && this.search(this.currentResultSet.beatmapsetIds.size)
                }

                restoreTurbolinks() {
                    this.restoreStateFromUrl(), this.search(0, !0), null != this.initialErrorMessage && (osu.popup(this.initialErrorMessage, "danger"), delete this.initialErrorMessage)
                }

                search(e = 0, t = !1) {
                    return d(this, void 0, void 0, (function* () {
                        if (this.isSupporterMissing || e < 0) return void (this.searchStatus = {
                            error: null,
                            from: e,
                            restore: t,
                            state: "completed"
                        });
                        let s;
                        this.searchStatus = {from: 0, restore: t, state: 0 === e ? "searching" : "paging"};
                        try {
                            yield this.beatmapsetSearch.get(this.filters, e)
                        } catch (r) {
                            s = 0 !== r.readyState ? r : null
                        }
                        Object(o.u)(() => {
                            this.searchStatus = {
                                error: s,
                                from: e,
                                restore: t,
                                state: "completed"
                            }, this.currentResultSet = this.beatmapsetSearch.getResultSet(this.filters)
                        })
                    }))
                }

                updateFilters(e) {
                    this.filters.update(e)
                }

                filterChangedSearch() {
                    const t = Object(i.a)("beatmapsets.index", this.filters.queryParams);
                    e.controller.advanceHistory(t), this.search()
                }

                restoreStateFromUrl() {
                    const e = Object(c.a)().href, t = BeatmapsetFilter.filtersFromUrl(e);
                    null != this.filtersObserver && this.filtersObserver(), this.filters = new n.a(e), this.filtersObserver = Object(o.r)(this.filters, this.filterChangedHandler), this.isExpanded = Object(a.intersection)(Object.keys(t), BeatmapsetFilter.expand).length > 0
                }
            }

            u([o.q], p.prototype, "advancedSearch", void 0), u([o.q], p.prototype, "currentResultSet", void 0), u([o.q], p.prototype, "filters", void 0), u([o.q], p.prototype, "isExpanded", void 0), u([o.q], p.prototype, "searchStatus", void 0), u([o.h], p.prototype, "currentBeatmapsetIds", null), u([o.h], p.prototype, "error", null), u([o.h], p.prototype, "hasMore", null), u([o.h], p.prototype, "isBusy", null), u([o.h], p.prototype, "isPaging", null), u([o.h], p.prototype, "isSupporterMissing", null), u([o.h], p.prototype, "recommendedDifficulty", null), u([o.h], p.prototype, "supporterRequiredFilterText", null), u([o.f], p.prototype, "cancel", null), u([o.f], p.prototype, "loadMore", null), u([o.f], p.prototype, "restoreTurbolinks", null), u([o.f], p.prototype, "search", null), u([o.f], p.prototype, "updateFilters", null), u([o.f], p.prototype, "restoreStateFromUrl", null)
        }).call(this, s("dMdw"))
    }, zr5c: function (e, t, s) {
        "use strict";
        s.d(t, "a", (function () {
            return u
        }));
        var r = s("sHNI"), n = s("/G9H"), i = s("tSlR"), a = s("dTpI"), o = s("TezV"), l = s("ss8h");
        const c = ["praise", "problem", "suggestion"];

        class u extends n.Component {
            constructor() {
                super(...arguments), Object.defineProperty(this, "select", {
                    enumerable: !0,
                    configurable: !0,
                    writable: !0,
                    value: e => {
                        const t = a.b.findPath(this.context, this.props.element);
                        i.h.setNodes(this.context, {discussionType: e}, {at: t})
                    }
                })
            }

            render() {
                const e = c.map(e => ({
                    icon: n.createElement("span", {className: `beatmap-discussion-message-type beatmap-discussion-message-type--${e}`}, n.createElement("i", {className: r.a[e]})),
                    id: e,
                    label: osu.trans(`beatmaps.discussions.message_type.${e}`)
                }));
                return n.createElement(o.a, {
                    disabled: this.props.disabled,
                    menuOptions: e,
                    onSelect: this.select,
                    selected: this.props.element.discussionType
                })
            }
        }

        Object.defineProperty(u, "contextType", {enumerable: !0, configurable: !0, writable: !0, value: l.a})
    }, zrLC: function (e, t, s) {
        "use strict";
        (function (e) {
            s.d(t, "a", (function () {
                return l
            }));
            var r = s("ueqr"), n = s("0h6b"), i = s("lv9K"), a = s("/jJF"), o = function (e, t, s, r) {
                var n, i = arguments.length, a = i < 3 ? t : null === r ? r = Object.getOwnPropertyDescriptor(t, s) : r;
                if ("object" == typeof Reflect && "function" == typeof Reflect.decorate) a = Reflect.decorate(e, t, s, r); else for (var o = e.length - 1; o >= 0; o--) (n = e[o]) && (a = (i < 3 ? n(a) : i > 3 ? n(t, s, a) : n(t, s)) || a);
                return i > 3 && a && Object.defineProperty(t, s, a), a
            };

            class l {
                constructor() {
                    Object.defineProperty(this, "current", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), Object.defineProperty(this, "updatingOptions", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: !1
                    }), Object.defineProperty(this, "user", {
                        enumerable: !0,
                        configurable: !0,
                        writable: !0,
                        value: void 0
                    }), this.current = Object.assign({}, r.a, this.fromStorage()), Object(i.p)(this)
                }

                get(e) {
                    return this.current[e]
                }

                set(t, s) {
                    if (this.current[t] !== s && (this.current[t] = s, localStorage.userPreferences = JSON.stringify(this.current), null != this.user)) return this.updatingOptions = !0, e.ajax(Object(n.a)("account.options"), {
                        data: {user_profile_customization: {[t]: s}},
                        dataType: "JSON",
                        method: "PUT"
                    }).done(t => {
                        e.publish("user:update", t)
                    }).fail(Object(a.c)()).always(() => {
                        this.updatingOptions = !1
                    })
                }

                setUser(e) {
                    this.user = e, null == e || this.updatingOptions || (this.current = null == e ? void 0 : e.user_preferences)
                }

                fromStorage() {
                    try {
                        const e = JSON.parse(localStorage.userPreferences);
                        if (null != e && "object" == typeof e) return e
                    } catch (e) {
                    }
                    return {}
                }
            }

            o([i.q], l.prototype, "current", void 0), o([i.f], l.prototype, "set", null), o([i.f], l.prototype, "setUser", null)
        }).call(this, s("5wds"))
    }
}]);
//# sourceMappingURL=commons.91e3ef36.js.map