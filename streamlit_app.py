# Energy under load
plt.figure()
plt.plot(t, E)
plt.axhline(E_TARGET, linestyle="--")
plt.title("Everlasting Battery: Energy bounded under load")
plt.xlabel("t")
plt.ylabel("E")
plt.show()

# K stays inside stable-flow band
plt.figure()
plt.plot(t, K_series)
plt.axhline(K_LOW, linestyle="--")
plt.axhline(K_FREEZE, linestyle="--")
plt.title("K regulated inside stable-flow window")
plt.xlabel("t")
plt.ylabel("K")
plt.show()

# Event density actuator
plt.figure()
plt.plot(t, f_m)
plt.title("Event density f_m (adaptive control)")
plt.xlabel("t")
plt.ylabel("f_m")
plt.show()