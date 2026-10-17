package AyoServer.Arc;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Golden-output harness (g-376-199): runs the env server's own ArcWorldBuilder, ArcGridGeometry and
 * ArcGridStats (compiled from the sources named in the fixture's meta) over the cases in args[0] and
 * writes what they return to args[1]. It adds nothing to the builder's output.
 */
public final class GoldenDump {
    private GoldenDump() {}

    public static void main(String[] args) throws Exception {
        JsonObject in = new JsonObject(Files.readString(Path.of(args[0])));
        JsonObject out = new JsonObject();

        JsonObject scopes = new JsonObject();
        for (Object id : in.getJsonArray("scope_ids")) {
            scopes.put((String) id, ArcWorldBuilder.scopeOf((String) id));
        }
        out.put("scopes", scopes);

        JsonArray cases = new JsonArray();
        for (Object o : in.getJsonArray("cases")) {
            JsonObject c = (JsonObject) o;
            String gameId = c.getString("game_id");
            String scope = ArcWorldBuilder.scopeOf(gameId);
            JsonArray cursor = c.getJsonArray("cursor");
            JsonArray units = ArcWorldBuilder.buildRootUnits(
                c.getJsonArray("grid"), cursor.getInteger(0), cursor.getInteger(1), scope);
            cases.add(new JsonObject()
                .put("name", c.getString("name"))
                .put("game_id", gameId)
                .put("scope", scope)
                .put("cursor", cursor)
                .put("grid", c.getJsonArray("grid"))
                .put("units", units));
        }
        out.put("cases", cases);

        JsonObject colors = new JsonObject();
        for (int m = 0; m <= 16; m++) {
            for (int v = 0; v <= 16; v++) {
                colors.put(v + "," + m, ArcGridGeometry.color(v, m));
            }
        }
        out.put("colors", colors);

        Files.writeString(Path.of(args[1]), out.encode());
    }
}
