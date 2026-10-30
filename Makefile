.DEFAULT_GOAL := package

.PHONY: package
package:
	install -d package/DEBIAN package/usr/lib/systemd/system package/home/rpy/Activities
	printf 'Package: activities\nVersion: 1.0.0\nArchitecture: all\nMaintainer: rpy <rpy@localhost>\nDescription: Activities display and feed watcher\n' > package/DEBIAN/control
	install -m 0644 services/activities.service services/watcher.service services/unclutter.service package/usr/lib/systemd/system/
	install -m 0644 activities.py drive.py watcher.py package/home/rpy/Activities/
	chmod 0755 package/DEBIAN/postinst
	dpkg-deb --root-owner-group --build package activities.deb